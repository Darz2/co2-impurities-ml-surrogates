# export JULIA_NUM_THREADS=2
# Usage:  julia Mixture_source1.jl [source_id]
# Default source_id = 1

SOURCE_ID = isempty(ARGS) ? 1 : parse(Int, ARGS[1])

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())
println("source_id             = ", SOURCE_ID)

using CSV
using DataFrames
using Random
using Statistics
using Plots
using Serialization
using SymbolicRegression
using JSON3

cd(@__DIR__)
outdir = "outputs/source$(SOURCE_ID)"
mkpath(outdir)

include("sr_utils.jl")

# ============================================================
# Critical properties from PC-SAFT + cDFT
# Tc [K],  Pc [bar]  →  loaded from Pure_fluid/component_Tc.json
# ============================================================

crit = JSON3.read(read("../Pure_fluid/component_Tc.json", String))

# Pc is stored in bar in the JSON; P in the CSV is in MPa → convert bar → MPa
Tc_map = Dict(string(k) => Float64(v["Tc"])       for (k, v) in crit if string(k) != "carbon_dioxide")
Pc_map = Dict(string(k) => Float64(v["Pc"]) / 10  for (k, v) in crit if string(k) != "carbon_dioxide")

# ============================================================
# Data loading
# ============================================================

df_all = CSV.read("1691761_CombinedDataset_A3.csv", DataFrame; normalizenames=true)
df     = filter(row -> row.source_id == SOURCE_ID, df_all)

println("Rows for source_id = $(SOURCE_ID): ", nrow(df))

# ============================================================
# CO2 baseline (WSD)
# ============================================================

r1             = (df[!, :rhoL_carbon_dioxide] .- df[!, :rhoV_carbon_dioxide]) ./
                 (df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide])
df[!, :gamma_base] = df[!, :gamma0_carbon_dioxide] .* r1 .^ 2
df[!, :gamma_cDFT] = df[!, :gamma_wsd] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]

# Target: relative residual of cDFT vs WSD baseline
df[!, :eps_base] = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

# ============================================================
# Feature engineering
# ============================================================

x_imp_cols = [c for c in names(df) if startswith(c, "x_") && c != "x_carbon_dioxide"]
y_imp_cols = [c for c in names(df) if startswith(c, "y_") && c != "y_carbon_dioxide"]
z_imp_cols = [c for c in names(df) if startswith(c, "z_") && c != "z_carbon_dioxide"]
imp_names  = replace.(x_imp_cols, "x_" => "")

println("Impurity components (", length(imp_names), "): ", join(imp_names, ", "))

missing_Tc = [n for n in imp_names if !haskey(Tc_map, n)]
missing_Pc = [n for n in imp_names if !haskey(Pc_map, n)]
isempty(missing_Tc) || error("Missing Tc for: $(join(missing_Tc, ", "))")
isempty(missing_Pc) || error("Missing Pc for: $(join(missing_Pc, ", "))")

eps_small = 1e-12

# Total impurity mole fractions
df[!, :x_total_impurity]     = reduce(.+, [df[!, c] for c in x_imp_cols])
df[!, :y_total_impurity]     = reduce(.+, [df[!, c] for c in y_imp_cols])
df[!, :z_total_impurity]     = reduce(.+, [df[!, c] for c in z_imp_cols])
df[!, :delta_total_impurity] = df[!, :x_total_impurity] .- df[!, :y_total_impurity]
df[!, :logK_total_impurity]  = log.((df[!, :y_total_impurity] .+ eps_small) ./
                                    (df[!, :x_total_impurity] .+ eps_small))

# Composition-weighted reduced T and P per impurity
# impurity_Tr_{z,x,y}  = Σᵢ φᵢ·(T/Tcᵢ)   (φ = z, x, or y mole fraction)
# impurity_Pr_{z,x,y}  = Σᵢ φᵢ·(P/Pcᵢ)
# _avg variants        = above ÷ total impurity fraction
# _delta variants      = Σᵢ (xᵢ−yᵢ)·(T/Tcᵢ or P/Pcᵢ)
for col in (:impurity_Tr_z, :impurity_Tr_x, :impurity_Tr_y,
            :impurity_Pr_z, :impurity_Pr_x, :impurity_Pr_y,
            :impurity_delta_Tr, :impurity_delta_Pr)
    df[!, col] = zeros(nrow(df))
end

for imp in imp_names
    Tr_i = df[!, :T] ./ Tc_map[imp]
    Pr_i = df[!, :P] ./ Pc_map[imp]
    xi, yi, zi = df[!, Symbol("x_"*imp)], df[!, Symbol("y_"*imp)], df[!, Symbol("z_"*imp)]

    df[!, :impurity_Tr_z] .+= zi .* Tr_i;  df[!, :impurity_Pr_z] .+= zi .* Pr_i
    df[!, :impurity_Tr_x] .+= xi .* Tr_i;  df[!, :impurity_Pr_x] .+= xi .* Pr_i
    df[!, :impurity_Tr_y] .+= yi .* Tr_i;  df[!, :impurity_Pr_y] .+= yi .* Pr_i
    df[!, :impurity_delta_Tr] .+= (xi .- yi) .* Tr_i
    df[!, :impurity_delta_Pr] .+= (xi .- yi) .* Pr_i
end

for (sum_col, tot_col, avg_col) in (
    (:impurity_Tr_z, :z_total_impurity, :impurity_Tr_avg_z),
    (:impurity_Tr_x, :x_total_impurity, :impurity_Tr_avg_x),
    (:impurity_Tr_y, :y_total_impurity, :impurity_Tr_avg_y),
    (:impurity_Pr_z, :z_total_impurity, :impurity_Pr_avg_z),
    (:impurity_Pr_x, :x_total_impurity, :impurity_Pr_avg_x),
    (:impurity_Pr_y, :y_total_impurity, :impurity_Pr_avg_y),
)
    df[!, avg_col] = df[!, sum_col] ./ (df[!, tot_col] .+ eps_small)
end

# ============================================================
# Target and feature selection
# ============================================================

target = :eps_base

features = [
    :z_total_impurity, :x_total_impurity, :y_total_impurity,
    :delta_total_impurity, :logK_total_impurity,
    :impurity_Tr_z,    :impurity_Tr_x,    :impurity_Tr_y,
    :impurity_delta_Tr,:impurity_Tr_avg_z,:impurity_Tr_avg_x, :impurity_Tr_avg_y,
    :impurity_Pr_z,    :impurity_Pr_x,    :impurity_Pr_y,
    :impurity_delta_Pr,:impurity_Pr_avg_z,:impurity_Pr_avg_x, :impurity_Pr_avg_y,
]

n_before = nrow(df)
dropmissing!(df, vcat(features, [target]))
filter!(row -> all(isfinite(row[c]) for c in vcat(features, [target])), df)
println("Rows after cleaning: $(nrow(df))  (dropped $(n_before - nrow(df)))")

println("\nTarget:   ", target)
println("Features: ", features)

X = Matrix{Float64}(df[:, features])
y = Vector{Float64}(df[:, target])

# ============================================================
# Train / validation / test split  (70 / 15 / 15)
# ============================================================

n   = size(X, 1)
rng = MersenneTwister(5002034)
idx = randperm(rng, n)

n_train = round(Int, 0.70 * n)
n_val   = round(Int, 0.15 * n)

train_idx = idx[1:n_train]
val_idx   = idx[n_train+1:n_train+n_val]
test_idx  = idx[n_train+n_val+1:end]

X_train, X_val, X_test = X[train_idx,:], X[val_idx,:], X[test_idx,:]
y_train, y_val, y_test = y[train_idx],   y[val_idx],   y[test_idx]

println("Train / Val / Test = $(size(X_train,1)) / $(size(X_val,1)) / $(size(X_test,1))")

# ============================================================
# Symbolic regression
# ============================================================

options = Options(
    binary_operators        = (+, -, *, safepow, safe_div),
    unary_operators         = (abs,),
    populations             = 80,
    maxsize                 = 16,
    parsimony               = 0.003f0,
    complexity_of_operators = [safe_div => 2, safepow => 3, abs => 2],
    complexity_of_constants = 2,
)

println("\n══ Fitting eps_base  (source_id = $(SOURCE_ID)) ══")

hof = equation_search(
    X_train', y_train;
    options        = options,
    niterations    = 200,
    variable_names = string.(features),
    parallelism    = :multithreading,
)

serialize("$outdir/hall_of_fame_eps_base.jls", hof)
println("Saved hall of fame → $outdir/hall_of_fame_eps_base.jls")

# ============================================================
# Select best equation
# ============================================================

dominating = calculate_pareto_frontier(hof)
select_best_equation(dominating, X_val, y_val, options)

best_idx = 8  # <-- CHANGE THIS INDEX BASED ON THE OUTPUT ABOVE
best_eq  = dominating[best_idx]

println("\nChosen index = $best_idx  |  $(string_tree(best_eq.tree, options))")

# ============================================================
# Evaluate and save predictions
# ============================================================

yhat_train, _ = eval_tree_array(best_eq.tree, X_train', options)
yhat_val,   _ = eval_tree_array(best_eq.tree, X_val',   options)
yhat_test,  _ = eval_tree_array(best_eq.tree, X_test',  options)

print_metrics("Train",      y_train, yhat_train)
print_metrics("Validation", y_val,   yhat_val)
print_metrics("Test",       y_test,  yhat_test)

CSV.write("$outdir/train_predictions_eps_base.csv",      DataFrame(Actual=y_train, Predicted=yhat_train))
CSV.write("$outdir/validation_predictions_eps_base.csv", DataFrame(Actual=y_val,   Predicted=yhat_val))
CSV.write("$outdir/test_predictions_eps_base.csv",       DataFrame(Actual=y_test,  Predicted=yhat_test))
println("Saved prediction CSVs → $outdir/")

# ============================================================
# Parity plot
# ============================================================

parity_plot(
    y_train, yhat_train,
    y_val,   yhat_val,
    y_test,  yhat_test;
    title    = "Parity Plot — ε  (source_id = $(SOURCE_ID))",
    xlabel   = "Actual ε",
    ylabel   = "Predicted ε",
    eq_idx   = best_idx,
    savepath = "$outdir/parity_plot_eps_base.png",
)
