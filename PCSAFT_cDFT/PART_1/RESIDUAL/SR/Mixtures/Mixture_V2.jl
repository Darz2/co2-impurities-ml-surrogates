# V2 — original compositional features only, with batching enabled.
# Excludes T, P, Tr, Pr, P/Psat0, density-gap, and gamma_base_ratio
# features (kept in V1 Mixture.jl).
#
# export JULIA_NUM_THREADS=2

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())

using CSV
using DataFrames
using Random
using Statistics
using Plots
using Serialization
using SymbolicRegression
using Parquet2
using JSON3

cd(@__DIR__)
const OUTDIR = "SR_MIXTURES_OUTPUTS_V2"
mkpath(OUTDIR)
include("sr_utils.jl")

mae(y, yhat) = mean(abs.(y .- yhat))

# safe_div appears in trained equations but is not handled by sr_utils.string_to_math
function expand_safe_div(eq_str::AbstractString)
    s = String(eq_str)
    for _ in 1:200
        result = extract_function_args(s, "safe_div", true)
        result === nothing && break
        arg1, arg2, start, end_pos = result
        arg1 = strip_outer_parens(arg1)
        arg2 = strip_outer_parens(arg2)
        s = s[1:start-1] * "(($arg1) / ($arg2))" * s[end_pos+1:end]
    end
    return s
end

# ============================================================
# Data loading and feature engineering
# ============================================================
df                  = CSV.read("../../CombinedDatasetSEC_A4.csv", DataFrame; normalizenames=true)
df[!, :gamma_base]  = df[!, :gamma_wsd_UC]
df[!, :gamma_cDFT]  = df[!, :gamma_wsd_UC] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]

# Target: relative residual vs baseline
df[!, :eps_base] = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

# ============================================================
# Dimensionless candidate features (composition only)
# ============================================================
x_imp_cols = [c for c in names(df) if startswith(c, "x_") && c != "x_carbon_dioxide"]
y_imp_cols = [c for c in names(df) if startswith(c, "y_") && c != "y_carbon_dioxide"]
z_imp_cols = [c for c in names(df) if startswith(c, "z_") && c != "z_carbon_dioxide"]

println("\nImpurity components (", length(x_imp_cols), "): ",
        join(replace.(x_imp_cols, "x_" => ""), ", "))

df[!, :x_total_impurity]     = reduce(.+, [df[!, c] for c in x_imp_cols])
df[!, :y_total_impurity]     = reduce(.+, [df[!, c] for c in y_imp_cols])
df[!, :z_total_impurity]     = reduce(.+, [df[!, c] for c in z_imp_cols])
df[!, :delta_total_impurity] = df[!, :x_total_impurity] .- df[!, :y_total_impurity]

eps = 1e-12
df[!, :logK_total_impurity] = log.((df[!, :y_total_impurity] .+ eps) ./ (df[!, :x_total_impurity] .+ eps))

# ============================================================
# Target and features for Symbolic Regression
# ============================================================
target = :eps_base

features = [
    # Composition aggregates (original 5)
    :z_total_impurity,
    :x_total_impurity,
    :y_total_impurity,
    :delta_total_impurity,
    :logK_total_impurity,
]

println("\nTarget used: ", target)
println("Features used: ", features)

n_before = nrow(df)
dropmissing!(df, vcat(features, [target]))
all_cols = vcat(features, [target])
filter!(row -> all(isfinite(row[c]) for c in all_cols), df)
n_after = nrow(df)
if n_before > n_after
    println("Dropped $(n_before - n_after) rows with missing/Inf/NaN values ($(n_after) remaining)")
end

X          = Matrix{Float64}(df[:, features])
y          = Vector{Float64}(df[:, target])
gamma_base = Vector{Float64}(df[:, :gamma_base])

# ============================================================
# Train/validation/test split
# ============================================================
n = size(X, 1)
rng = MersenneTwister(5002034)
idx = randperm(rng, n)

n_train = round(Int, 0.70 * n)
n_val   = round(Int, 0.15 * n)

train_idx = idx[1:n_train]
val_idx   = idx[n_train+1:n_train+n_val]
test_idx  = idx[n_train+n_val+1:end]

X_train, X_val, X_test = X[train_idx, :], X[val_idx, :], X[test_idx, :]
y_train, y_val, y_test = y[train_idx],    y[val_idx],    y[test_idx]
gamma_base_train = gamma_base[train_idx]
gamma_base_val   = gamma_base[val_idx]
gamma_base_test  = gamma_base[test_idx]

println("Training rows   = ", size(X_train, 1))
println("Validation rows = ", size(X_val, 1))
println("Testing rows    = ", size(X_test, 1))

# ============================================================
# Symbolic regression options
# ============================================================
options = Options(
    binary_operators        = (+, -, *, safepow, safe_div),
    unary_operators         = (abs,),
    populations             = 80,
    maxsize                 = 16,
    parsimony               = 0.003f0,
    complexity_of_operators = [safe_div => 2, safepow => 3, abs => 2],
    complexity_of_constants = 2,
    batching                = true,
    batch_size              = 1000,
)

# ============================================================
# Train symbolic regression
# ============================================================
println("\n══ Fitting residual correction eps_base (V2: composition + density gap) ══")

hall_of_fame_eps_base = equation_search(
    X_train', y_train;
    options        = options,
    niterations    = 200,
    variable_names = string.(features),
    parallelism    = :multithreading,
)

serialize(joinpath(OUTDIR, "hall_of_fame_eps_base.jls"), hall_of_fame_eps_base)
println("\nSaved hall of fame to: ", joinpath(OUTDIR, "hall_of_fame_eps_base.jls"))
println("\nHall of fame summary:")
println(hall_of_fame_eps_base)

# ============================================================
# Choose best equation using validation RMSE
# ============================================================
dominating_eps_base = calculate_pareto_frontier(hall_of_fame_eps_base)
best_idx_auto, best_rmse_val_auto = select_best_equation(dominating_eps_base, X_val, y_val, options)

manual_best_idx = 3
best_idx = manual_best_idx === nothing ? best_idx_auto : manual_best_idx
println("\nUsing best_idx = $best_idx  (auto-pick = $best_idx_auto, val RMSE = $best_rmse_val_auto)")

best_eq_eps_base = dominating_eps_base[best_idx]
best_eq_str = string_tree(best_eq_eps_base.tree, options)

println("\nChosen equation index = ", best_idx)
println("Chosen equation:")
println(best_eq_eps_base)
println("\nEquation string:")
println(best_eq_str)

# ============================================================
# Evaluate chosen equation
# ============================================================
yhat_train, ok_train = eval_tree_array(best_eq_eps_base.tree, X_train', options)
yhat_val,   ok_val   = eval_tree_array(best_eq_eps_base.tree, X_val',   options)
yhat_test,  ok_test  = eval_tree_array(best_eq_eps_base.tree, X_test',  options)

println("\nEvaluation successful (train): ", ok_train)
println("Evaluation successful (val):   ", ok_val)
println("Evaluation successful (test):  ", ok_test)

print_metrics("Train", y_train, yhat_train)
print_metrics("Validation", y_val, yhat_val)
print_metrics("Test", y_test, yhat_test)

rmse_train = rmse(y_train, yhat_train)
rmse_val   = rmse(y_val,   yhat_val)
rmse_test  = rmse(y_test,  yhat_test)

r2_train = r2_score(y_train, yhat_train)
r2_val   = r2_score(y_val,   yhat_val)
r2_test  = r2_score(y_test,  yhat_test)

mae_train = mae(y_train, yhat_train)
mae_val   = mae(y_val,   yhat_val)
mae_test  = mae(y_test,  yhat_test)

# Reconstructed gamma_cDFT predictions (mN/m)
gamma_cDFT_actual_train = gamma_base_train .* (1.0 .+ y_train)
gamma_cDFT_actual_val   = gamma_base_val   .* (1.0 .+ y_val)
gamma_cDFT_actual_test  = gamma_base_test  .* (1.0 .+ y_test)

gamma_cDFT_pred_train = gamma_base_train .* (1.0 .+ yhat_train)
gamma_cDFT_pred_val   = gamma_base_val   .* (1.0 .+ yhat_val)
gamma_cDFT_pred_test  = gamma_base_test  .* (1.0 .+ yhat_test)

gamma_rmse_train = rmse(gamma_cDFT_actual_train, gamma_cDFT_pred_train)
gamma_rmse_val   = rmse(gamma_cDFT_actual_val,   gamma_cDFT_pred_val)
gamma_rmse_test  = rmse(gamma_cDFT_actual_test,  gamma_cDFT_pred_test)

gamma_r2_train = r2_score(gamma_cDFT_actual_train, gamma_cDFT_pred_train)
gamma_r2_val   = r2_score(gamma_cDFT_actual_val,   gamma_cDFT_pred_val)
gamma_r2_test  = r2_score(gamma_cDFT_actual_test,  gamma_cDFT_pred_test)

gamma_mae_train = mae(gamma_cDFT_actual_train, gamma_cDFT_pred_train)
gamma_mae_val   = mae(gamma_cDFT_actual_val,   gamma_cDFT_pred_val)
gamma_mae_test  = mae(gamma_cDFT_actual_test,  gamma_cDFT_pred_test)

# ============================================================
# Save predictions (incl. reconstructed gamma_cDFT for cross-model overlay)
# ============================================================
CSV.write(joinpath(OUTDIR, "SR_train_predictions_eps_base.csv"),
    DataFrame(
        Actual            = y_train,
        Predicted         = yhat_train,
        gamma_base        = gamma_base_train,
        gamma_cDFT_actual = gamma_cDFT_actual_train,
        gamma_cDFT_pred   = gamma_cDFT_pred_train,
    ))
CSV.write(joinpath(OUTDIR, "SR_validation_predictions_eps_base.csv"),
    DataFrame(
        Actual            = y_val,
        Predicted         = yhat_val,
        gamma_base        = gamma_base_val,
        gamma_cDFT_actual = gamma_cDFT_actual_val,
        gamma_cDFT_pred   = gamma_cDFT_pred_val,
    ))
CSV.write(joinpath(OUTDIR, "SR_test_predictions_eps_base.csv"),
    DataFrame(
        Actual            = y_test,
        Predicted         = yhat_test,
        gamma_base        = gamma_base_test,
        gamma_cDFT_actual = gamma_cDFT_actual_test,
        gamma_cDFT_pred   = gamma_cDFT_pred_test,
    ))

println("\nSaved prediction tables in $OUTDIR")

# ============================================================
# Save metrics JSON (schema compatible with GPR/SVGP loaders)
# ============================================================
best_complexity = compute_complexity(best_eq_eps_base, options)
best_loss       = best_eq_eps_base.loss

eq_raw   = best_eq_str
eq_math  = string_to_math(expand_safe_div(eq_raw))
eq_latex = math_to_latex(eq_math)

metrics = Dict(
    "model"                   => "SR_eps_base_V2",
    "target"                  => String(target),
    "features"                => String.(features),
    "seed"                    => 5002034,
    "n_train"                 => length(y_train),
    "n_val"                   => length(y_val),
    "n_test"                  => length(y_test),
    "train_rmse"              => rmse_train,
    "val_rmse"                => rmse_val,
    "test_rmse"               => rmse_test,
    "train_r2"                => r2_train,
    "val_r2"                  => r2_val,
    "test_r2"                 => r2_test,
    "train_mae"               => mae_train,
    "val_mae"                 => mae_val,
    "test_mae"                => mae_test,
    "gamma_cDFT_rmse_train"   => gamma_rmse_train,
    "gamma_cDFT_rmse_val"     => gamma_rmse_val,
    "gamma_cDFT_rmse_test"    => gamma_rmse_test,
    "gamma_cDFT_r2_train"     => gamma_r2_train,
    "gamma_cDFT_r2_val"       => gamma_r2_val,
    "gamma_cDFT_r2_test"      => gamma_r2_test,
    "gamma_cDFT_mae_train_mNm"=> gamma_mae_train,
    "gamma_cDFT_mae_val_mNm"  => gamma_mae_val,
    "gamma_cDFT_mae_test_mNm" => gamma_mae_test,
    "equation_index"          => best_idx,
    "equation_raw"            => eq_raw,
    "equation_math"           => eq_math,
    "equation_latex"          => eq_latex,
    "complexity"              => best_complexity,
    "sr_loss"                 => best_loss,
)

open(joinpath(OUTDIR, "SR_residual_metrics.json"), "w") do io
    JSON3.pretty(io, metrics)
end

println("\nSaved metrics JSON: ", joinpath(OUTDIR, "SR_residual_metrics.json"))

# ============================================================
# Quick parity plot (Plots.jl) — full publication plot is done by the .ipynb
# ============================================================
xmin = min(
    minimum(y_train), minimum(yhat_train),
    minimum(y_val),   minimum(yhat_val),
    minimum(y_test),  minimum(yhat_test),
)
xmax = max(
    maximum(y_train), maximum(yhat_train),
    maximum(y_val),   maximum(yhat_val),
    maximum(y_test),  maximum(yhat_test),
)

p_eps_base = scatter(
    y_train, yhat_train;
    xlabel = "Actual ε / [-]",
    ylabel = "Predicted ε / [-]",
    label  = "Train",
    markersize = 4,
    title  = "Parity Plot for ε (V2)",
    legend = :topleft,
)
scatter!(p_eps_base, y_val,  yhat_val;  label = "Validation", markersize = 4)
scatter!(p_eps_base, y_test, yhat_test; label = "Test",       markersize = 4)
plot!(p_eps_base, [xmin, xmax], [xmin, xmax]; label = "y = x",
      lw = 2, color = :black, line = :dash)

annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.72*(xmax - xmin),
    text("Chosen Eq. index = $best_idx", 9, :left))
annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.78*(xmax - xmin),
    text("Train: RMSE = $(round(rmse_train, sigdigits=2)), R² = $(round(r2_train, sigdigits=2))", 9, :left))
annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.84*(xmax - xmin),
    text("Val: RMSE = $(round(rmse_val, sigdigits=2)), R² = $(round(r2_val, sigdigits=2))", 9, :left))
annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.90*(xmax - xmin),
    text("Test: RMSE = $(round(rmse_test, sigdigits=2)), R² = $(round(r2_test, sigdigits=2))", 9, :left))

savefig(p_eps_base, joinpath(OUTDIR, "SR_parity_plot_eps_base.png"))
savefig(p_eps_base, joinpath(OUTDIR, "SR_parity_plot_eps_base.pdf"))
println("\nSaved parity plot: ", joinpath(OUTDIR, "SR_parity_plot_eps_base.png"))
