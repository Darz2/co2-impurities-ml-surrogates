# dump_equations_and_augment_V1.jl
# V1 counterpart of dump_equations_and_augment.jl, for SR_MIXTURES_OUTPUTS.
# (a) Dump the full Pareto hall of fame to a readable CSV.
# (b) Rewrite the SR prediction CSVs with the input feature columns added,
#     so the equation is verifiable per-row.
#
# Run from this directory:   julia dump_equations_and_augment_V1.jl

using CSV
using DataFrames
using Random
using Statistics
using Serialization
using SymbolicRegression
using JSON3

cd(@__DIR__)
const OUTDIR = "SR_MIXTURES_OUTPUTS"
include("sr_utils.jl")

mae(y, yhat) = mean(abs.(y .- yhat))

# Must match the operators used during training in Mixture.jl (V1)
function training_options()
    return Options(
        binary_operators        = (+, -, *, safepow, safe_div),
        unary_operators         = (abs, safe_sqrt, safe_log),
        complexity_of_operators = [
            safe_div  => 2,
            safepow   => 3,
            abs       => 2,
            safe_sqrt => 2,
            safe_log  => 2,
        ],
        complexity_of_constants = 2,
    )
end

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
# (a) Dump the full hall of fame
# ============================================================
options = training_options()
hof = deserialize(joinpath(OUTDIR, "hall_of_fame_eps_base.jls"))
dominating = calculate_pareto_frontier(hof)

println("\n══ Full Pareto frontier (", length(dominating), " equations) ══")
rows = NamedTuple[]
for (i, member) in enumerate(dominating)
    eq_raw  = string_tree(member.tree, options)
    eq_math = string_to_math(expand_safe_div(eq_raw))
    eq_ltx  = math_to_latex(eq_math)
    cplx    = compute_complexity(member, options)
    println("[$i] complexity=$cplx  loss=$(member.loss)")
    println("     raw : ", eq_raw)
    println("     math: ", eq_math)
    push!(rows, (index=i, complexity=cplx, loss=member.loss,
                 equation_raw=eq_raw, equation_math=eq_math, equation_latex=eq_ltx))
end

eq_df = DataFrame(rows)
CSV.write(joinpath(OUTDIR, "hall_of_fame_equations.csv"), eq_df)
println("\nWrote ", joinpath(OUTDIR, "hall_of_fame_equations.csv"))

# ============================================================
# (b) Rebuild data + identical split, add input features to prediction CSVs
# ============================================================
df                  = CSV.read("../../CombinedDatasetSEC_A4.csv", DataFrame; normalizenames=true)
df[!, :gamma_base]  = df[!, :gamma_wsd_UC]
df[!, :gamma_cDFT]  = df[!, :gamma_wsd_UC] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]
df[!, :eps_base]    = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

# Composition aggregates
x_imp_cols = [c for c in names(df) if startswith(c, "x_") && c != "x_carbon_dioxide"]
y_imp_cols = [c for c in names(df) if startswith(c, "y_") && c != "y_carbon_dioxide"]
z_imp_cols = [c for c in names(df) if startswith(c, "z_") && c != "z_carbon_dioxide"]
df[!, :x_total_impurity]     = reduce(.+, [df[!, c] for c in x_imp_cols])
df[!, :y_total_impurity]     = reduce(.+, [df[!, c] for c in y_imp_cols])
df[!, :z_total_impurity]     = reduce(.+, [df[!, c] for c in z_imp_cols])
df[!, :delta_total_impurity] = df[!, :x_total_impurity] .- df[!, :y_total_impurity]
eps = 1e-12
df[!, :logK_total_impurity]  = log.((df[!, :y_total_impurity] .+ eps) ./ (df[!, :x_total_impurity] .+ eps))

# Physics-informed features (reduced by pure-CO2 references)
df[!, :Tr]               = df[!, :T] ./ df[!, :Tc_carbon_dioxide]
df[!, :oneMinusTr]       = 1.0 .- df[!, :Tr]
df[!, :Pr]               = df[!, :P] ./ df[!, :Pc_carbon_dioxide]
df[!, :P_over_Psat0]     = df[!, :P] ./ df[!, :Psat0_carbon_dioxide]
df[!, :delta_rho]        = df[!, :rho_l] .- df[!, :rho_v]
df[!, :delta_rho_norm]   = (df[!, :rho_l] .- df[!, :rho_v]) ./
                          ((df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide]) .+ eps)
df[!, :gamma_base_ratio] = df[!, :gamma_wsd_UC] ./ (df[!, :gamma0_carbon_dioxide] .+ eps)

target   = :eps_base
features = [
    :z_total_impurity, :x_total_impurity, :y_total_impurity,
    :delta_total_impurity, :logK_total_impurity,
    :Tr, :oneMinusTr, :Pr, :P_over_Psat0,
    :delta_rho, :delta_rho_norm, :gamma_base_ratio,
]

dropmissing!(df, vcat(features, [target]))
all_cols = vcat(features, [target])
filter!(row -> all(isfinite(row[c]) for c in all_cols), df)

X          = Matrix{Float64}(df[:, features])
y          = Vector{Float64}(df[:, target])
gamma_base = Vector{Float64}(df[:, :gamma_base])

# IDENTICAL split (same seed as Mixture.jl)
n   = size(X, 1)
rng = MersenneTwister(5002034)
idx = randperm(rng, n)
n_train = round(Int, 0.70 * n)
n_val   = round(Int, 0.15 * n)
train_idx = idx[1:n_train]
val_idx   = idx[n_train+1:n_train+n_val]
test_idx  = idx[n_train+n_val+1:end]

# Chosen equation = first Pareto member at TARGET_COMPLEXITY = 12 (as in Mixture.jl)
TARGET_COMPLEXITY = 12
best_idx = findfirst(m -> compute_complexity(m, options) == TARGET_COMPLEXITY, dominating)
@assert best_idx !== nothing "no Pareto member at complexity $TARGET_COMPLEXITY"
best_eq  = dominating[best_idx]
println("\nUsing chosen equation index = $best_idx (complexity $TARGET_COMPLEXITY): ",
        string_tree(best_eq.tree, options))

function augment(split_idx, fname)
    Xs   = X[split_idx, :]
    ys   = y[split_idx]
    gb   = gamma_base[split_idx]
    yhat, ok = eval_tree_array(best_eq.tree, Xs', options)
    @assert ok "evaluation failed for $fname"
    out = DataFrame(Xs, Symbol.(features))      # input feature columns
    out.Actual            = ys
    out.Predicted         = yhat
    out.gamma_base        = gb
    out.gamma_cDFT_actual = gb .* (1.0 .+ ys)
    out.gamma_cDFT_pred   = gb .* (1.0 .+ yhat)
    path = joinpath(OUTDIR, fname)
    CSV.write(path, out)
    println("Wrote $path  (", nrow(out), " rows, cols: ", join(names(out), ", "), ")")
end

augment(train_idx, "SR_train_predictions_eps_base_with_features.csv")
augment(val_idx,   "SR_validation_predictions_eps_base_with_features.csv")
augment(test_idx,  "SR_test_predictions_eps_base_with_features.csv")

println("\nDone.")
