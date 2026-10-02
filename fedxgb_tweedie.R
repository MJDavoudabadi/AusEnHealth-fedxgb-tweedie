# =============================================================================
# fedxgb_tweedie.R
#
# Federated histogram-based XGBoost with a Tweedie objective, for predicting
# a non-negative, right-skewed, zero-inflated outcome (here, mortality rate)
# across multiple data-holding clients (here, Australian states/territories),
# benchmarked against an equivalent centralised model trained on pooled data.
#
# This is a "horizontal FL" simulator: every client shares the same feature
# set but holds a different, non-overlapping set of observations. It runs as
# a single-machine simulation - all clients' data live on one disk - but the
# code is deliberately structured so that no function ever needs more than a
# client's own raw rows plus small, already-aggregated numbers from other
# clients. That structure is what would let this be deployed across genuinely
# separate machines with no code changes to the aggregation logic itself -
# only the in-process client loop would need to become network calls.
#
# WHAT MAKES THIS FEDERATED, NOT JUST DISTRIBUTED COMPUTE
# ---------------------------------------------------------------------------
# A tree ensemble can be "federated" in several ways (e.g. training separate
# per-client models and averaging their outputs). This script instead
# federates gradient boosting itself, at the level of a single split
# decision, so the resulting model is built from exactly the same
# information a centralised model would use - just computed in pieces:
#
#   - Categorical levels: each client scans its own train.csv and reduces
#     it to its set of unique category labels; only that small label set
#     (never a repeated, row-level column) is combined across clients
#     (get_global_categorical_levels()).
#
#   - Histogram bin edges: each client computes its own local quantiles
#     from its own training data; the server merges by averaging each
#     client's corresponding quantile value. Only B numbers per client
#     cross client boundaries - never a raw feature value
#     (make_global_bin_edges_federated()).
#
#   - Split finding: at every node of every tree, each client computes a
#     local gradient/Hessian histogram from only its own rows currently
#     assigned to that node; these per-client histograms are combined by
#     simple element-wise summation before the best split is chosen
#     (build_node_histograms_for_client(), aggregate_node_histograms()).
#     Because splitting a sum across clients and adding the pieces back
#     together reproduces the same total as computing it in one place,
#     this recovers the same split decisions a centralised model would
#     make from the same (approximately binned) data - it is not an
#     approximation of gradient boosting, just a distributed way of
#     computing the same sums.
#
#   - Evaluation metrics (NLL, RMSSE, MASE): each client reduces its own
#     held-out predictions to a handful of sufficient statistics - a row
#     count and a few sums - and only those numbers are aggregated to
#     report a pooled or per-client metric (compute_client_test_stats(),
#     metrics_from_stats(), naive_scale_stats()). Because these metrics are
#     themselves sums divided by a count, summing client-level statistics
#     before dividing reproduces the exact pooled figure without ever
#     pooling an individual client's row-level predictions.
#
#   - Early stopping: the federated model's early-stopping criterion is a
#     client-size-weighted average of each client's own validation loss,
#     which reproduces the pooled validation loss a centralised model's
#     built-in early stopping would compute - using only a row count and
#     an already-local loss value per client, never raw validation rows.
#
# What this script does NOT implement: encryption, secure aggregation, or
# differential privacy on top of the above - clients here compute their own
# local statistics honestly, and the "server" is trusted to aggregate them
# correctly. It also does not implement a real network layer; the client
# loop is a plain R for()-loop over in-memory list elements, not RPC calls
# to separate machines. Both would be required for a genuinely
# privacy-hardened, multi-party deployment; this script implements the
# statistical/algorithmic structure that such a deployment would sit on top
# of - what information is ever allowed to leave a client - not the
# network or cryptographic security layer around it.
#
# One diagnostic exception, called out again at the point it occurs: a few
# visualisation-only plots near the end of the script (a residual histogram
# and a predicted-vs-actual scatter) do pool every client's individual
# predictions into one in-memory table purely to draw those specific
# figures. That object is never used by anything in the training or metrics
# pipeline above it - it exists only because this is a single-machine
# simulator being used for model debugging, not because federation requires
# it. In a genuinely distributed deployment, those two plots would either be
# rendered locally per client or omitted from any pooled report.
#
# INPUT DATA
# ---------------------------------------------------------------------------
# Expects one folder per client under data_root, each containing three CSVs
# from a chronological three-way split: train.csv (grows trees), es_valid.csv
# (early-stopping round selection only - never used for final metrics), and
# test.csv (touched once, at the end, for all reported metrics/plots).
# =============================================================================

suppressPackageStartupMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
  if (!requireNamespace("xgboost",    quietly = TRUE)) install.packages("xgboost")
  if (!requireNamespace("ggplot2",    quietly = TRUE)) install.packages("ggplot2")
  if (!requireNamespace("gridExtra",  quietly = TRUE)) install.packages("gridExtra")
  library(data.table)
  library(xgboost)
  library(ggplot2)
  library(gridExtra)
})

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# Real dataset root
# data_root <- "DATA_ROOT"

# Simulated dataset root
data_root <-"DATA_ROOT"

output_dir <- "fl_output_agg_lags_3way"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

y_col <- "mortality_rate"

x_cols <- c(
  "age_group",
  "tmean",
  "erp",
  "IRSAD",
  "mortality_lag1",
  "mortality_lag2",
  "mortality_lag12",
  "tmean_lag1",
  "tmean_lag2",
  "tmean_lag12"
)

B <- 256L
max_depth <- 5L
nrounds <- 300L
early_stopping_rounds <- 30L
eta <- 0.05
lambda_l2 <- 1.0
sampling_fraction <- 0.5
sampling_method <- "none"
set.seed(42)

# Tweedie parameter
tweedie_variance_power <- 1.5
eps <- 1e-10

# =============================================================================
# 2. HELPERS
# =============================================================================

tweedie_nll_terms <- function(y, mu, rho = tweedie_variance_power) {
  mu <- pmax(mu, eps)
  - y * mu^(1 - rho) / (1 - rho) + mu^(2 - rho) / (2 - rho)
}

tweedie_nloglik <- function(y, mu, rho = tweedie_variance_power) {
  mean(tweedie_nll_terms(y, mu, rho))
}

# -----------------------------------------------------------------------------
# RMSSE / MASE naive scale, from mortality_lag1 on TRAINING data only, per
# client (state); this mirrors the report convention: naive_j =
# mortality_lag1_j.
#
# naive_scale_stats(): run LOCALLY by each state on its own TRAINING rows.
# Reduces them to three numbers - n, sum of squared naive error, sum of
# absolute naive error - and returns NOTHING row-level.
#
# naive_scale_from_stats(): run on the "server" (or by a single client on
# its own stats row). Turns (n, sum_sq_err, sum_abs_err) into (mse, mae).
# Summing several states' stats before calling this reproduces the exact
# pooled naive denominator - so, just like the TEST metrics above, the
# national RMSSE/MASE denominator never requires pooling individual
# training observations.
# -----------------------------------------------------------------------------
naive_scale_stats <- function(y_train, naive_train) {
  data.table(
    n           = length(y_train),
    sum_sq_err  = sum((y_train - naive_train)^2),
    sum_abs_err = sum(abs(y_train - naive_train))
  )
}

naive_scale_from_stats <- function(stats) {
  list(
    mse = stats$sum_sq_err / stats$n,
    mae = stats$sum_abs_err / stats$n
  )
}

# -----------------------------------------------------------------------------
# FEDERATED-SAFE METRIC AGGREGATION.
#
# compute_client_test_stats(): run LOCALLY by each state on its own TEST
# rows. Reduces them to four numbers - n, sum of Tweedie NLL terms, sum of
# squared error, sum of absolute error - and returns NOTHING row-level.
# These four numbers are the only thing that ever crosses to the "server".
#
# metrics_from_stats(): run on the "server" (or by a single client on its
# own stats row). Turns (n, sum_nll, sum_sq_err, sum_abs_err) plus a naive
# scale (also just two numbers, from local training data) into NLL / RMSSE /
# MASE. Because sums are additive, summing several states' stats before
# calling this reproduces the exact pooled metric - so a correct global
# number never requires pooling individual test outcomes.
# -----------------------------------------------------------------------------
compute_client_test_stats <- function(y, yhat, rho) {
  data.table(
    n           = length(y),
    sum_nll     = sum(tweedie_nll_terms(y, yhat, rho)),
    sum_sq_err  = sum((y - yhat)^2),
    sum_abs_err = sum(abs(y - yhat))
  )
}

metrics_from_stats <- function(stats, naive_mse, naive_mae) {
  data.table(
    n     = stats$n,
    nll   = stats$sum_nll / stats$n,
    rmsse = sqrt(stats$sum_sq_err / stats$n) / sqrt(naive_mse),
    mase  = (stats$sum_abs_err / stats$n) / naive_mae
  )
}

calc_grad_hess_tweedie <- function(y, f, rho = tweedie_variance_power) {
  f <- pmin(pmax(f, -20), 20)
  
  g <- -y * exp((1 - rho) * f) + exp((2 - rho) * f)
  h <- -y * (1 - rho) * exp((1 - rho) * f) +
    (2 - rho) * exp((2 - rho) * f)
  
  h <- pmax(h, eps)
  
  list(g = g, h = h)
}

leaf_weight <- function(G, H, lambda_l2, eta) {
  -eta * G / (H + lambda_l2)
}

split_gain <- function(GL, HL, GR, HR, lambda_l2) {
  sc <- function(G, H) G^2 / (H + lambda_l2)
  0.5 * (sc(GL, HL) + sc(GR, HR) - sc(GL + GR, HL + HR))
}

uniform_mask <- function(n, frac) {
  if (frac >= 1) return(rep(TRUE, n))
  k <- max(1L, round(frac * n))
  idx <- sample.int(n, k, replace = FALSE)
  out <- rep(FALSE, n)
  out[idx] <- TRUE
  out
}

# -----------------------------------------------------------------------------
# Federated (non-pooled) histogram bin edges.
# Each client computes its OWN local quantiles from its OWN training data
# only. The "server" merges by averaging each client's corresponding
# quantile value across clients. Only B numbers per client are ever
# communicated - raw rows never leave a client. This approximates (but does
# not exactly reproduce) the true pooled-data quantiles.
# -----------------------------------------------------------------------------
make_global_bin_edges_federated <- function(client_data_list, x_cols, B) {
  edges <- vector("list", length(x_cols))
  names(edges) <- x_cols
  probs <- seq(0, 1, length.out = B + 1L)[-c(1L, B + 1L)]
  
  for (feat in x_cols) {
    local_quantiles <- lapply(client_data_list, function(ld) {
      x <- as.numeric(ld$train[[feat]])
      x <- x[is.finite(x)]
      if (length(x) == 0L) return(NULL)
      as.numeric(stats::quantile(x, probs = probs, na.rm = TRUE, type = 7))
    })
    local_quantiles <- local_quantiles[!vapply(local_quantiles, is.null, logical(1))]
    
    if (length(local_quantiles) == 0L) {
      stop(sprintf("Feature '%s' has no finite values in any client's training data.", feat))
    }
    
    # Server-side merge: simple average of each client's corresponding
    # quantile. (A size-weighted average is an easy variant if client
    # training-set sizes are considered acceptable to share - sizes are far
    # less sensitive than raw feature values.)
    merged <- Reduce(`+`, local_quantiles) / length(local_quantiles)
    cuts <- sort(unique(merged[is.finite(merged)]))
    edges[[feat]] <- c(-Inf, cuts, Inf)
  }
  edges
}

to_bin_idx <- function(x, edges) {
  findInterval(x, vec = edges, rightmost.closed = TRUE, all.inside = TRUE)
}

bin_data <- function(dt, x_cols, global_edges) {
  out <- as.data.table(lapply(x_cols, function(feat) to_bin_idx(dt[[feat]], global_edges[[feat]])))
  setnames(out, x_cols)
  out
}

predict_one_tree <- function(tree, bin_row) {
  nid <- "1"
  repeat {
    node <- tree$nodes[[nid]]
    if (is.null(node)) return(0.0)
    if (isTRUE(node$is_leaf)) return(as.numeric(node$weight))
    b <- as.integer(bin_row[[node$feature]])
    nid <- if (b <= node$split_bin) as.character(node$left_id) else as.character(node$right_id)
  }
}

predict_margin <- function(model, X_bins) {
  pred <- rep(model$base_margin, nrow(X_bins))
  if (length(model$trees) == 0L) return(pred)
  
  for (tree in model$trees) {
    pred <- pred + vapply(
      seq_len(nrow(X_bins)),
      function(i) predict_one_tree(tree, X_bins[i, , drop = FALSE]),
      FUN.VALUE = 0.0
    )
  }
  pred
}

predict_response <- function(model, X_bins) {
  exp(pmin(pmax(predict_margin(model, X_bins), -20), 20))
}

# =============================================================================
# 3. DATA LOADING AND ONE-HOT ENCODING (3-way: train / es_valid / test)
# =============================================================================

categorical_x_cols <- intersect(c("age_group", "sex", "cause"), x_cols)
numeric_x_cols     <- setdiff(x_cols, categorical_x_cols)

# -----------------------------------------------------------------------------
# Only train.csv is read here. es_valid/test files are never touched when
# building categorical levels, so validation and test labels can never
# influence encoding.
# -----------------------------------------------------------------------------
get_global_categorical_levels <- function(client_dirs, categorical_cols) {
  out <- vector("list", length(categorical_cols))
  names(out) <- categorical_cols
  
  for (cc in categorical_cols) {
    # Client-side: each state reduces its own train.csv column to just its
    # unique category labels before anything is combined - a client's full,
    # repeated, row-level column is never what gets shared onward.
    per_client_levels <- lapply(client_dirs, function(client_dir) {
      tr <- fread(file.path(client_dir, "train.csv"), select = cc)
      unique(as.character(tr[[cc]]))
    })
    # Server-side: union of small unique-label sets only.
    vals <- unlist(per_client_levels, use.names = FALSE)
    out[[cc]] <- sort(unique(vals[!is.na(vals)]))
  }
  out
}

# -----------------------------------------------------------------------------
# Encodes train, es_valid, and test together (so factor levels/dummy columns
# line up across all three) and returns all three consistently encoded.
# -----------------------------------------------------------------------------
encode_for_xgb <- function(train, es_valid, test, y_col, x_cols, categorical_levels) {
  categorical_cols <- names(categorical_levels)
  numeric_cols     <- setdiff(x_cols, categorical_cols)
  keep_cols        <- c(y_col, x_cols)
  
  train    <- train[, ..keep_cols]
  es_valid <- es_valid[, ..keep_cols]
  test     <- test[, ..keep_cols]
  
  train[, split := "train"]
  es_valid[, split := "es_valid"]
  test[, split := "test"]
  all_data <- rbindlist(list(train, es_valid, test), fill = TRUE)
  
  for (nm in c(y_col, numeric_cols)) {
    all_data[, (nm) := as.numeric(get(nm))]
  }
  
  for (cc in categorical_cols) {
    all_data[, (cc) := factor(as.character(get(cc)),
                              levels = categorical_levels[[cc]])]
  }
  
  all_data <- all_data[complete.cases(all_data[, c(y_col, x_cols, "split"), with = FALSE])]
  
  rhs <- paste(x_cols, collapse = " + ")
  form <- as.formula(paste("~", rhs, "- 1"))
  mm <- model.matrix(form, data = all_data)
  
  encoded <- as.data.table(mm)
  encoded[, (y_col) := all_data[[y_col]]]
  encoded[, split := all_data$split]
  
  train_encoded    <- encoded[split == "train"]
  es_valid_encoded <- encoded[split == "es_valid"]
  test_encoded     <- encoded[split == "test"]
  
  train_encoded[, split := NULL]
  es_valid_encoded[, split := NULL]
  test_encoded[, split := NULL]
  
  encoded_x_cols <- setdiff(names(train_encoded), y_col)
  
  list(
    train = train_encoded,
    es_valid = es_valid_encoded,
    test = test_encoded,
    x_cols = encoded_x_cols
  )
}

load_one_client <- function(client_dir, y_col, x_cols, categorical_levels) {
  train    <- fread(file.path(client_dir, "train.csv"))
  es_valid <- fread(file.path(client_dir, "es_valid.csv"))
  test     <- fread(file.path(client_dir, "test.csv"))
  
  encode_for_xgb(
    train = train,
    es_valid = es_valid,
    test = test,
    y_col = y_col,
    x_cols = x_cols,
    categorical_levels = categorical_levels
  )
}

# =============================================================================
# 4. NODE-SPECIFIC CLIENT HISTOGRAMS
# =============================================================================

build_node_histograms_for_client <- function(client_state, node_id, x_cols, B) {
  sel <- client_state$node_assign == node_id
  if (!any(sel)) return(NULL)
  
  g <- client_state$g[sel]
  h <- client_state$h[sel]
  X <- client_state$bins[sel, , drop = FALSE]
  
  hists <- vector("list", length(x_cols))
  names(hists) <- x_cols
  
  for (feat in x_cols) {
    bins_f <- X[[feat]]
    
    GH <- rowsum(cbind(G = g, H = h), bins_f, reorder = TRUE)
    
    G_full <- numeric(B)
    H_full <- numeric(B)
    
    bin_ids <- as.integer(rownames(GH))
    
    G_full[bin_ids] <- GH[, "G"]
    H_full[bin_ids] <- GH[, "H"]
    
    hists[[feat]] <- list(G = G_full, H = H_full)
  }
  
  hists
}

aggregate_node_histograms <- function(client_hist_list, x_cols, B) {
  non_null <- client_hist_list[!vapply(client_hist_list, is.null, logical(1))]
  if (length(non_null) == 0L) return(NULL)
  
  out <- vector("list", length(x_cols))
  names(out) <- x_cols
  for (feat in x_cols) {
    G <- numeric(B)
    H <- numeric(B)
    for (hh in non_null) {
      G <- G + hh[[feat]]$G
      H <- H + hh[[feat]]$H
    }
    out[[feat]] <- list(G = G, H = H)
  }
  out
}

find_best_split <- function(agg_hists, x_cols, B, lambda_l2) {
  feat0 <- x_cols[1L]
  G_tot <- sum(agg_hists[[feat0]]$G)
  H_tot <- sum(agg_hists[[feat0]]$H)
  
  best_gain <- -Inf
  best_feat <- NULL
  best_bin <- NULL
  
  for (feat in x_cols) {
    G_bins <- agg_hists[[feat]]$G
    H_bins <- agg_hists[[feat]]$H
    G_cum <- cumsum(G_bins)
    H_cum <- cumsum(H_bins)
    
    for (k in seq_len(B - 1L)) {
      GL <- G_cum[k]; HL <- H_cum[k]
      GR <- G_tot - GL; HR <- H_tot - HL
      if (HL <= 0 || HR <= 0) next
      gain <- split_gain(GL, HL, GR, HR, lambda_l2)
      if (is.finite(gain) && gain > best_gain) {
        best_gain <- gain
        best_feat <- feat
        best_bin <- k
      }
    }
  }
  
  list(
    gain = best_gain,
    feature = best_feat,
    split_bin = best_bin,
    G_tot = G_tot,
    H_tot = H_tot
  )
}

# =============================================================================
# 5. FIT ONE FEDERATED TREE (uses TRAIN rows only - unchanged)
# =============================================================================

fit_one_fl_tree <- function(model, client_data_list, x_cols, B, max_depth,
                            eta, lambda_l2, sampling_method, sampling_fraction) {
  
  client_round_states <- lapply(client_data_list, function(ld) {
    f_hat <- predict_margin(model, ld$train_bins)
    gh <- calc_grad_hess_tweedie(ld$y_train, f_hat, tweedie_variance_power)
    
    mask <- switch(
      sampling_method,
      "uniform" = uniform_mask(length(gh$g), sampling_fraction),
      "none"    = rep(TRUE, length(gh$g)),
      stop("Unknown sampling_method: ", sampling_method)
    )
    if (!any(mask)) mask[sample.int(length(mask), 1L)] <- TRUE
    
    list(
      bins = ld$train_bins[mask, , drop = FALSE],
      g = gh$g[mask],
      h = gh$h[mask],
      node_assign = rep(1L, sum(mask))
    )
  })
  
  nodes <- list()
  next_id <- 2L
  frontier <- list(list(node_id = 1L, depth = 0L))
  
  while (length(frontier) > 0L) {
    cur <- frontier[[1L]]
    frontier <- frontier[-1L]
    
    node_id <- cur$node_id
    depth <- cur$depth
    
    client_hists <- lapply(client_round_states, build_node_histograms_for_client,
                           node_id = node_id, x_cols = x_cols, B = B)
    
    agg_hists <- aggregate_node_histograms(client_hists, x_cols, B)
    
    if (is.null(agg_hists)) {
      nodes[[as.character(node_id)]] <- list(is_leaf = TRUE, weight = 0.0)
      next
    }
    
    split <- find_best_split(agg_hists, x_cols, B, lambda_l2)
    G_tot <- split$G_tot
    H_tot <- split$H_tot
    
    if (depth >= max_depth || !is.finite(split$gain) || split$gain <= 0 || H_tot <= 0) {
      nodes[[as.character(node_id)]] <- list(
        is_leaf = TRUE,
        weight = leaf_weight(G_tot, H_tot, lambda_l2, eta)
      )
      next
    }
    
    left_id <- next_id; next_id <- next_id + 1L
    right_id <- next_id; next_id <- next_id + 1L
    
    nodes[[as.character(node_id)]] <- list(
      is_leaf = FALSE,
      feature = split$feature,
      split_bin = split$split_bin,
      left_id = left_id,
      right_id = right_id
    )
    
    for (i in seq_along(client_round_states)) {
      st <- client_round_states[[i]]
      in_node <- st$node_assign == node_id
      if (!any(in_node)) next
      
      feat_bins <- st$bins[[split$feature]][in_node]
      left_mask <- feat_bins <= split$split_bin
      
      idx_node <- which(in_node)
      st$node_assign[idx_node[left_mask]] <- left_id
      st$node_assign[idx_node[!left_mask]] <- right_id
      client_round_states[[i]] <- st
    }
    
    frontier[[length(frontier) + 1L]] <- list(node_id = left_id, depth = depth + 1L)
    frontier[[length(frontier) + 1L]] <- list(node_id = right_id, depth = depth + 1L)
  }
  
  list(nodes = nodes)
}

# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

client_dirs <- sort(list.dirs(data_root, full.names = TRUE, recursive = FALSE))
if (length(client_dirs) == 0L) stop("No client folders found in: ", data_root)

categorical_levels <- get_global_categorical_levels(client_dirs, categorical_x_cols)

client_data_list <- vector("list", length(client_dirs))
names(client_data_list) <- basename(client_dirs)

for (i in seq_along(client_dirs)) {
  ld <- load_one_client(
    client_dirs[i],
    y_col = y_col,
    x_cols = x_cols,
    categorical_levels = categorical_levels
  )
  client_data_list[[i]] <- ld
}

x_cols <- client_data_list[[1L]]$x_cols

# Federated (non-pooled) bin edges - see helper above.
global_edges <- make_global_bin_edges_federated(client_data_list, x_cols, B)

for (nm in names(client_data_list)) {
  ld <- client_data_list[[nm]]
  ld$train_bins    <- bin_data(ld$train, x_cols, global_edges)
  ld$es_valid_bins <- bin_data(ld$es_valid, x_cols, global_edges)
  ld$test_bins     <- bin_data(ld$test, x_cols, global_edges)
  ld$y_train    <- ld$train[[y_col]]
  ld$y_es_valid <- ld$es_valid[[y_col]]
  ld$y_test     <- ld$test[[y_col]]
  client_data_list[[nm]] <- ld
}

train_sizes <- vapply(client_data_list, function(ld) nrow(ld$train), numeric(1))
train_means <- vapply(client_data_list, function(ld) mean(ld$y_train), numeric(1))

# Per-state es_valid row counts - just a count per state, needed to weight
# FL early stopping the same way the centralised run's pooled es_valid set
# implicitly weights every row.
es_valid_sizes <- vapply(client_data_list, function(ld) nrow(ld$es_valid), numeric(1))

base_mu <- sum(train_sizes * train_means) / sum(train_sizes)
base_mu <- pmax(base_mu, eps)

model <- list(
  base_margin = log(base_mu),
  trees = list(),
  x_cols = x_cols,
  bin_edges = global_edges,
  tweedie_variance_power = tweedie_variance_power
)

best_model <- model
best_tweedie <- Inf
patience <- 0L
history <- numeric(nrounds)

cat(sprintf("Clients: %d | Features: %d | B=%d | max_depth=%d | sampling=%s | S=%.2f | Tweedie power=%.2f\n",
            length(client_data_list), length(x_cols), B, max_depth,
            toupper(sampling_method), sampling_fraction, tweedie_variance_power))

for (r in seq_len(nrounds)) {
  tree_r <- fit_one_fl_tree(
    model = model,
    client_data_list = client_data_list,
    x_cols = x_cols,
    B = B,
    max_depth = max_depth,
    eta = eta,
    lambda_l2 = lambda_l2,
    sampling_method = sampling_method,
    sampling_fraction = sampling_fraction
  )
  model$trees[[length(model$trees) + 1L]] <- tree_r
  
  # Early stopping uses es_valid only - never the final reporting (test) set.
  es_valid_tweedies <- vapply(client_data_list, function(ld) {
    pred <- predict_response(model, ld$es_valid_bins)
    tweedie_nloglik(ld$y_es_valid, pred, tweedie_variance_power)
  }, numeric(1))
  
  # Weight each state's ES-valid NLL by its own es_valid row count
  # (es_valid_sizes) so this reproduces the pooled ES-valid NLL a real
  # server could compute from just (n_j, NLL_j) per state - matching how
  # XGBoost's built-in early stopping on the pooled centralised es_valid
  # set implicitly weights every row, not every state, equally. Only a
  # count and a state's own already-local NLL are combined here - no
  # row-level validation data crosses to this aggregation.
  avg_tweedie <- sum(es_valid_sizes * es_valid_tweedies) / sum(es_valid_sizes)
  history[r] <- avg_tweedie
  
  cat(sprintf("Round %3d | pooled ES-valid Tweedie NLL (state-size-weighted) = %.6f | trees = %d\n",
              r, avg_tweedie, length(model$trees)))
  
  if (avg_tweedie < best_tweedie) {
    best_tweedie <- avg_tweedie
    best_model <- model
    patience <- 0L
  } else {
    patience <- patience + 1L
  }
  
  if (patience >= early_stopping_rounds) {
    cat(sprintf("Early stopping at round %d\n", r))
    break
  }
}

model <- best_model

# =============================================================================
# 7. FINAL FL METRICS - FEDERATED-SAFE
#    Computed ONLY on the held-out test set, which was never used for
#    tree-growing or for early-stopping/round-selection. Every number below
#    is derived from client-local sufficient statistics - no table of
#    individual (y_true, y_pred) TEST pairs is ever pooled.
# =============================================================================

# ---- 7a. Client-side: local prediction + local reduction to four numbers ---
fl_client_test_stats <- rbindlist(lapply(names(client_data_list), function(nm) {
  ld    <- client_data_list[[nm]]
  pred  <- predict_response(model, ld$test_bins)   # prediction computed locally, from the shared FL model
  stats <- compute_client_test_stats(ld$y_test, pred, tweedie_variance_power)
  stats[, client := nm]
  stats
}))
setcolorder(fl_client_test_stats, c("client", "n", "sum_nll", "sum_sq_err", "sum_abs_err"))

# ---- 7b. Server-side: only ever touches the four summary numbers per state -
fl_global_stats <- fl_client_test_stats[, .(
  n           = sum(n),
  sum_nll     = sum(sum_nll),
  sum_sq_err  = sum(sum_sq_err),
  sum_abs_err = sum(sum_abs_err)
)]

fl_tweedie_pooled <- fl_global_stats$sum_nll / fl_global_stats$n

cat(sprintf("\nFinal pooled FL TEST Tweedie NLL: %.6f\n", fl_tweedie_pooled))

# Only the sufficient-statistics table is written out - it's safe to share
# (aggregate counts and sums per state), unlike a row-level predictions file.
fwrite(fl_client_test_stats, file.path(output_dir, "fl_client_test_sufficient_stats.csv"))
saveRDS(model, file.path(output_dir, "fl_xgboost_model.rds"))

history_used <- history[history > 0]
training_curve_dt <- data.table(
  round = seq_along(history_used),
  avg_es_valid_tweedie_nll = history_used
)
fwrite(training_curve_dt, file.path(output_dir, "fl_training_curve.csv"))

# =============================================================================
# 8. CENTRALISED HIST-XGBOOST BASELINE
#    xgb.train's built-in early stopping uses es_valid; final metrics are
#    computed only on test - the same 3-way discipline as the FL model above.
# =============================================================================

central_train    <- rbindlist(lapply(client_data_list, function(ld) ld$train))
central_es_valid <- rbindlist(lapply(client_data_list, function(ld) ld$es_valid))

# Keep a per-row state label on the test set (train/es_valid don't need one -
# they're only ever used pooled) so centralised TEST predictions can later be
# broken down per state, the same way the FL predictions already are.
central_test <- rbindlist(lapply(names(client_data_list), function(nm) {
  dt <- copy(client_data_list[[nm]]$test)
  dt[, client := nm]
  dt
}))

dtrain    <- xgb.DMatrix(as.matrix(central_train[, ..x_cols]),    label = central_train[[y_col]])
des_valid <- xgb.DMatrix(as.matrix(central_es_valid[, ..x_cols]), label = central_es_valid[[y_col]])
dtest     <- xgb.DMatrix(as.matrix(central_test[, ..x_cols]),     label = central_test[[y_col]])

central_model <- xgb.train(
  params = list(
    objective              = "reg:tweedie",
    eval_metric            = paste0("tweedie-nloglik@", tweedie_variance_power),
    tweedie_variance_power = tweedie_variance_power,
    eta                    = eta,
    max_depth              = max_depth,
    lambda                 = lambda_l2,
    tree_method            = "hist",
    max_bin                = B,
    nthread                = 1L
  ),
  data                  = dtrain,
  nrounds               = nrounds,
  evals                 = list(es_valid = des_valid),
  early_stopping_rounds = early_stopping_rounds,
  verbose               = 0L
)

central_pred <- predict(central_model, dtest)

central_tweedie <- tweedie_nloglik(
  central_test[[y_col]],
  central_pred,
  tweedie_variance_power
)

cat(sprintf("Centralised hist XGBoost TEST Tweedie NLL: %.6f\n", central_tweedie))

central_preds_dt <- data.table(
  client   = central_test$client,
  y_true   = central_test[[y_col]],
  y_pred   = central_pred,
  residual = central_test[[y_col]] - central_pred
)
fwrite(central_preds_dt, file.path(output_dir, "central_test_predictions.csv"))
saveRDS(central_model, file.path(output_dir, "centralised_xgboost_model.rds"))

# Centralised per-state / pooled sufficient statistics, using the same
# reduction as the FL side (section 7), so the two are combined identically
# below. Note this is NOT a privacy fix for the centralised model - by
# definition it already trained on everyone's pooled raw data - it's just
# reused here for a consistent, simple comparison-table pipeline.
central_client_test_stats <- central_preds_dt[, {
  s <- compute_client_test_stats(y_true, y_pred, tweedie_variance_power)
  as.list(s)
}, by = client]

central_global_stats <- compute_client_test_stats(
  central_preds_dt$y_true, central_preds_dt$y_pred, tweedie_variance_power
)

# =============================================================================
# 8b. PER-STATE COMPARISON: FL vs CENTRALISED (NLL, RMSSE, MASE)
#     Built entirely from sufficient statistics: each state's row comes from
#     its own (n, sum_nll, sum_sq_err, sum_abs_err) - computed locally in
#     section 7 for FL, and via the same reduction for centralised - plus
#     its own locally-reduced naive-scale stats (naive_scale_by_state below).
#     The ALL (pooled) row comes from SUMMING those same per-state numbers,
#     never from a table of individual test or training rows.
# =============================================================================

naive_scale_by_state <- rbindlist(lapply(names(client_data_list), function(nm) {
  tr <- client_data_list[[nm]]$train
  st <- naive_scale_stats(tr[[y_col]], tr$mortality_lag1)
  ns <- naive_scale_from_stats(st)
  data.table(client = nm, n = st$n, sum_sq_err = st$sum_sq_err, sum_abs_err = st$sum_abs_err,
             naive_mse = ns$mse, naive_mae = ns$mae)
}))

# Global naive denominator built by SUMMING the per-state sufficient
# statistics above - n_j, SSE^naive_j, SAE^naive_j - never by pooling raw
# training rows across states.
naive_scale_all_stats <- naive_scale_by_state[, .(
  n = sum(n), sum_sq_err = sum(sum_sq_err), sum_abs_err = sum(sum_abs_err)
)]
naive_scale_all <- naive_scale_from_stats(naive_scale_all_stats)

per_state_comparison <- rbindlist(lapply(names(client_data_list), function(nm) {
  ns    <- naive_scale_by_state[client == nm]
  fl_s  <- fl_client_test_stats[client == nm]
  cen_s <- central_client_test_stats[client == nm]
  
  fl_m  <- metrics_from_stats(fl_s,  ns$naive_mse, ns$naive_mae)
  cen_m <- metrics_from_stats(cen_s, ns$naive_mse, ns$naive_mae)
  
  data.table(
    state         = nm,
    n_test        = fl_m$n,
    fl_nll        = fl_m$nll,        central_nll   = cen_m$nll,
    fl_rmsse      = fl_m$rmsse,      central_rmsse = cen_m$rmsse,
    fl_mase       = fl_m$mase,       central_mase  = cen_m$mase
  )
}))

# Pooled ("ALL") row - matches fl_tweedie_pooled / central_tweedie above,
# plus their RMSSE/MASE counterparts, built by summing the per-state
# sufficient statistics computed in sections 7 and 8 (not by pooling rows).
all_fl_m  <- metrics_from_stats(fl_global_stats,      naive_scale_all$mse, naive_scale_all$mae)
all_cen_m <- metrics_from_stats(central_global_stats, naive_scale_all$mse, naive_scale_all$mae)

per_state_comparison <- rbind(
  per_state_comparison,
  data.table(
    state    = "ALL (pooled)",
    n_test   = all_fl_m$n,
    fl_nll   = all_fl_m$nll,   central_nll   = all_cen_m$nll,
    fl_rmsse = all_fl_m$rmsse, central_rmsse = all_cen_m$rmsse,
    fl_mase  = all_fl_m$mase,  central_mase  = all_cen_m$mase
  )
)

cat("\n=== FL vs Centralised - per-state TEST metrics (NLL / RMSSE / MASE) ===\n")
print(per_state_comparison)

fwrite(per_state_comparison, file.path(output_dir, "fl_vs_central_per_state_metrics.csv"))

# ---- 8c. Per-state RMSSE/MASE bar chart, FL vs Centralised ------------------
per_state_long <- rbindlist(list(
  per_state_comparison[, .(state, method = "Federated",   rmsse = fl_rmsse,      mase = fl_mase)],
  per_state_comparison[, .(state, method = "Centralised", rmsse = central_rmsse, mase = central_mase)]
))
per_state_long <- melt(per_state_long, id.vars = c("state", "method"),
                       measure.vars = c("rmsse", "mase"),
                       variable.name = "metric", value.name = "value")
per_state_long[, metric := toupper(metric)]
per_state_long[, state := factor(state, levels = c(sort(names(client_data_list)), "ALL (pooled)"))]

p_state_compare <- ggplot(per_state_long, aes(x = state, y = value, fill = method)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = c("Federated" = "#4477AA", "Centralised" = "#EE6677")) +
  labs(
    title    = "Per-State TEST Metrics: Federated vs Centralised XGBoost",
    subtitle = "Naive scale (mortality_lag1) computed from that state's own training data",
    x        = NULL, y = "Score (lower is better)", fill = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(plot.title      = element_text(face = "bold", hjust = 0.5),
        plot.subtitle   = element_text(colour = "grey40", hjust = 0.5),
        axis.text.x     = element_text(angle = 45, hjust = 1),
        legend.position = "top")

# =============================================================================
# 9. VISUALISATIONS (all now based on TEST set predictions)
# =============================================================================

cat("\nGenerating visualisation plots...\n")

save_plot <- function(p, fname, w = 10, h = 7) {
  ggsave(file.path(output_dir, fname), plot = p, width = w, height = h, dpi = 150)
  cat("  Saved:", fname, "\n")
}

subtitle_params <- sprintf(
  "B=%d | depth=%d | eta=%.3f | Tweedie power=%.2f",
  B, max_depth, eta, tweedie_variance_power
)

save_plot(p_state_compare, "plot_per_state_fl_vs_central.png",
          w = 9, h = 8)

p_curve <- ggplot(training_curve_dt, aes(x = round, y = avg_es_valid_tweedie_nll)) +
  geom_line(colour = "#2b7bba", linewidth = 0.8) +
  geom_point(colour = "#2b7bba", size = 1.2, alpha = 0.7) +
  labs(
    title    = "FL XGBoost: Training Curve",
    subtitle = paste0(subtitle_params, " | selection metric: ES-valid"),
    x        = "Boosting Round",
    y        = "Avg ES-Valid Tweedie NLL (across clients)"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(colour = "grey40", hjust = 0.5))
save_plot(p_curve, "plot_training_curve.png")

# Per-state FL TEST Tweedie NLL, derived directly from the sufficient
# statistics computed in section 7 (no row-level table needed for this).
fl_client_nll <- fl_client_test_stats[, .(client, test_tweedie_nll = sum_nll / n, n_test = n)]

p_mse <- ggplot(fl_client_nll,
                aes(x = reorder(client, test_tweedie_nll),
                    y = test_tweedie_nll,
                    fill = test_tweedie_nll)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "#74c69d", high = "#d62728") +
  labs(title = "TEST Tweedie NLL per Client", x = "Client", y = "Tweedie NLL") +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))
save_plot(p_mse, "plot_client_tweedie_nll.png", w = 8, h = 5)

# -----------------------------------------------------------------------------
# DIAGNOSTIC-ONLY: pooled row-level FL predictions, for VISUALISATION below.
#
# Nothing above this point (fl_client_test_stats, fl_global_stats,
# per_state_comparison, fl_client_nll, fl_tweedie_pooled) depends on
# individual test rows ever being pooled. The residual histogram and
# predicted-vs-actual scatter below are a convenience of running every
# "client" in one R session on one machine; a genuinely distributed
# deployment would either render these locally per state, or not produce a
# pooled version of them at all. None of the reported metrics use this
# object.
#
# Also attaches each row's age_group label, recovered from the one-hot
# encoded age_group dummy column(s) produced by encode_for_xgb()'s
# model.matrix() step. Because age_group is the FIRST term in a
# no-intercept formula, it gets full dummy encoding (one column per level,
# not the usual n-1), so this normally finds two columns, e.g.
# "age_group75-84"/"age_group85+" - detected here rather than hardcoded,
# since the exact column names depend on the literal category strings in
# the data.
# -----------------------------------------------------------------------------
age_group_cols <- grep("^age_group", names(client_data_list[[1]]$test), value = TRUE)
if (length(age_group_cols) == 0L) {
  stop("No age_group dummy column found in encoded test data - check categorical_x_cols/x_cols.")
}

get_age_group_label <- function(test_dt, age_group_cols) {
  if (length(age_group_cols) == 1L) {
    # n-1 dummy coding (only reachable if age_group isn't the first term):
    # 1 = the non-reference level named by the column; 0 = reference level,
    # whose exact string isn't recoverable from the encoded data alone.
    lvl <- sub("^age_group", "", age_group_cols)
    ifelse(test_dt[[age_group_cols]] == 1, lvl, paste0("not ", lvl))
  } else {
    # Full dummy coding (the expected case here): exactly one column is 1
    # per row: take whichever one that is.
    lvls   <- sub("^age_group", "", age_group_cols)
    mm_sub <- as.matrix(test_dt[, ..age_group_cols])
    lvls[max.col(mm_sub, ties.method = "first")]
  }
}

diag_pooled_fl_test_pred <- rbindlist(lapply(names(client_data_list), function(nm) {
  ld   <- client_data_list[[nm]]
  pred <- predict_response(model, ld$test_bins)
  data.table(
    client    = nm,
    age_group = get_age_group_label(ld$test, age_group_cols),
    y_true    = ld$y_test,
    y_pred    = pred,
    residual  = ld$y_test - pred
  )
}))

p_resid <- ggplot(diag_pooled_fl_test_pred, aes(x = residual)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60,
                 fill = "#4477AA", colour = "white", alpha = 0.85) +
  geom_vline(xintercept = 0, colour = "red", linetype = "dashed", linewidth = 0.8) +
  labs(
    title    = "Distribution of Residuals (TEST Set, All Clients)",
    subtitle = "Federated XGBoost predictions",
    x        = "Residual (y_true - y_pred)",
    y        = "Density"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))
save_plot(p_resid, "plot_residual_histogram.png")

p_scatter <- ggplot(diag_pooled_fl_test_pred, aes(x = y_pred, y = y_true, colour = client)) +
  geom_point(alpha = 0.45, size = 1.4) +
  geom_abline(slope = 1, intercept = 0, colour = "black",
              linetype = "dashed", linewidth = 0.8) +
  labs(
    title    = "Predicted vs Actual (TEST Set, All Clients)",
    subtitle = "Points on the dashed line = perfect prediction",
    x        = "Predicted", y = "Actual", colour = "Client"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title      = element_text(face = "bold", hjust = 0.5),
        legend.position = if (length(client_data_list) <= 8) "right" else "none")
save_plot(p_scatter, "plot_predicted_vs_actual.png")

compare_dt <- rbind(
  data.table(method = "Federated XGBoost",
             residual = diag_pooled_fl_test_pred$residual),
  data.table(method = "Centralised XGBoost (hist)",
             residual = central_preds_dt$residual)
)

p_compare <- ggplot(compare_dt, aes(x = residual, fill = method, colour = method)) +
  geom_density(alpha = 0.35, linewidth = 0.8) +
  geom_vline(xintercept = 0, colour = "black", linetype = "dashed") +
  scale_fill_manual(values = c(
    "Federated XGBoost"           = "#4477AA",
    "Centralised XGBoost (hist)"  = "#EE6677"
  )) +
  scale_colour_manual(values = c(
    "Federated XGBoost"           = "#4477AA",
    "Centralised XGBoost (hist)"  = "#EE6677"
  )) +
  labs(
    title    = "Residual Distribution: Federated vs Centralised XGBoost (TEST)",
    subtitle = sprintf("FL Tweedie NLL=%.3f | Central Tweedie NLL=%.3f | B=%d ",
                       fl_tweedie_pooled, central_tweedie, B),
    x        = "Residual (y_true - y_pred)",
    y        = "Density",
    fill     = NULL, colour = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5), legend.position = "top")
save_plot(p_compare, "plot_fl_vs_central_residuals.png")

cat("\nComputing final-round gradient histograms for visualisation...\n")

final_root_hists <- lapply(client_data_list, function(ld) {
  f_hat <- predict_margin(model, ld$train_bins)
  gh    <- calc_grad_hess_tweedie(ld$y_train, f_hat, tweedie_variance_power)
  
  mask <- switch(
    sampling_method,
    "uniform" = uniform_mask(length(gh$g), sampling_fraction),
    "none"    = rep(TRUE, length(gh$g))
  )
  if (!any(mask)) mask[1L] <- TRUE
  
  g_s    <- gh$g[mask]
  h_s    <- gh$h[mask]
  bins_s <- ld$train_bins[mask, , drop = FALSE]
  
  lapply(setNames(x_cols, x_cols), function(feat) {
    b      <- bins_s[[feat]]
    B_feat <- length(model$bin_edges[[feat]]) - 1L
    G_bins <- numeric(B_feat); H_bins <- numeric(B_feat)
    for (k in seq_len(B_feat)) {
      sel <- b == k
      G_bins[k] <- sum(g_s[sel]); H_bins[k] <- sum(h_s[sel])
    }
    list(G = G_bins, H = H_bins)
  })
})

agg_root_hist <- lapply(setNames(x_cols, x_cols), function(feat) {
  B_feat <- length(model$bin_edges[[feat]]) - 1L
  G_agg  <- numeric(B_feat); H_agg <- numeric(B_feat)
  for (ch in final_root_hists) {
    G_agg <- G_agg + ch[[feat]]$G
    H_agg <- H_agg + ch[[feat]]$H
  }
  list(G = G_agg, H = H_agg)
})

hist_plot_data <- rbindlist(lapply(x_cols, function(feat) {
  G_vec <- agg_root_hist[[feat]]$G
  B_f   <- length(G_vec)
  if (B_f > 40L) {
    grp   <- ceiling(seq_len(B_f) / (B_f / 40))
    G_vec <- as.numeric(tapply(G_vec, grp, sum))
    B_f   <- length(G_vec)
  }
  data.table(feature = feat, bin = seq_len(B_f), G = G_vec)
}))

p_ghist <- ggplot(hist_plot_data, aes(x = bin, y = G)) +
  geom_col(fill = "#6A5ACD", alpha = 0.8, width = 0.9) +
  facet_wrap(~ feature, scales = "free_y") +
  labs(
    title    = "Aggregated Gradient Histogram (Root Node, Final Round)",
    subtitle = sprintf("B=%d bins (display: up to 40) | %d features | Sampling: %s",
                       B, length(x_cols), toupper(sampling_method)),
    x        = "Bin Index",
    y        = "Sum of Gradients (G)"
  ) +
  theme_bw(base_size = 10) +
  theme(plot.title   = element_text(face = "bold", hjust = 0.5),
        strip.text   = element_text(face = "bold", hjust = 0.5),
        axis.text.x  = element_blank(),
        axis.ticks.x = element_blank())

n_feats <- length(x_cols)
save_plot(p_ghist, "plot_gradient_histograms.png",
          w = 12, h = max(4, 2 * ceiling(n_feats / 4)))

p_actual_density_by_age <- ggplot(diag_pooled_fl_test_pred,
                                  aes(x = y_true, fill = age_group, colour = age_group)) +
  geom_density(alpha = 0.4, linewidth = 0.8) +
  labs(
    title = "Distribution of Actual Mortality Rate by Age Group (TEST Set)",
    x     = "Actual mortality rate (y_true)",
    y     = "Density",
    fill  = "Age group", colour = "Age group"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "top")
save_plot(p_actual_density_by_age, "plot_actual_density_by_age_group.png")

p_pred_density_by_age <- ggplot(diag_pooled_fl_test_pred,
                                aes(x = y_pred, fill = age_group, colour = age_group)) +
  geom_density(alpha = 0.4, linewidth = 0.8) +
  labs(
    title    = "Distribution of Predicted Mortality Rate by Age Group (TEST Set)",
    subtitle = "Federated model - checking whether predicted bands correspond to age group",
    x        = "Predicted mortality rate (y_pred)",
    y        = "Density",
    fill     = "Age group", colour = "Age group"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, colour = "grey40"),
        legend.position = "top")
save_plot(p_pred_density_by_age, "plot_predicted_density_by_age_group.png")

# =============================================================================
# Relative importance for the central model
# =============================================================================

importance_matrix <- xgb.importance(colnames(dtrain), model = central_model)
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("  FINAL SUMMARY (all metrics on held-out TEST set only)\n")
cat("============================================================\n")
cat(sprintf("  FL XGBoost  pooled TEST Tweedie NLL       = %.6f\n", fl_tweedie_pooled))
cat(sprintf("  FL XGBoost  per-client mean TEST Tweedie NLL    = %.6f\n", mean(fl_client_nll$test_tweedie_nll)))
cat(sprintf("  Centralised TEST Tweedie NLL (hist, B=%d) = %.6f\n", B, central_tweedie))
cat(sprintf("  FL trees built: %d\n", length(model$trees)))
cat(sprintf("  Per-state FL vs Centralised (NLL/RMSSE/MASE): %s\n",
            file.path(output_dir, "fl_vs_central_per_state_metrics.csv")))
cat(sprintf("  All outputs saved to: %s\n", output_dir))
cat("============================================================\n")
cat("\nDone.\n")
