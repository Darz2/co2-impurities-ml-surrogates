using CSV, DataFrames, Random, Statistics
using SymbolicRegression
using Serialization

cd(@__DIR__)

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())

# =========================
# Load dataset
# =========================
df = CSV.read("CSV_OLD/interfacial_results_cleaned.csv", DataFrame)

target  = ["P_bubble", "P_dew"]
exclude = vcat(
    ["gamma", "interfacial_thickness", "E_carbon dioxide", "E_argon", "E_hydrogen"],
    ["x_carbon dioxide", "x_argon", "x_hydrogen", "y_carbon dioxide", "y_argon", "y_hydrogen"],
    ["z_carbon dioxide"],
    ["liquid_density", "vapor_density"],
    ["pressure"],
    target,
)

features = [col for col in names(df) if col ∉ exclude]
println("Features: ", features)

X = Matrix{Float64}(df[:, features])
y_bubble = Vector{Float64}(df[:, "P_bubble"])
y_dew    = Vector{Float64}(df[:, "P_dew"])

# =========================
# Custom operators
# =========================
square(x::T) where {T} = x * x
cube(x::T) where {T} = x * x * x
inv_(x::T) where {T} = one(T) / x
invsquare(x::T) where {T} = one(T) / (x * x)
safesqrt(x::T) where {T} = x >= zero(T) ? sqrt(x) : T(NaN)
safelog(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)

# =========================
# Shared options
# =========================
options = Options(;
    binary_operators = (+, -, *, /),
    unary_operators  = (square, cube, inv_, invsquare, safesqrt, safelog, abs),
    populations      = 40,
    maxsize          = 30,
    parsimony        = 0.001f0,
)

# =========================
# Metrics
# =========================
rmse(y, yhat) = sqrt(mean((y .- yhat).^2))

function r2_score(y, yhat)
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    return 1 - ss_res / ss_tot
end

# =========================
# Build 5 folds
# =========================
function make_kfold_indices(n::Int, k::Int; seed::Int=85)
    idx = randperm(MersenneTwister(seed), n)
    fold_sizes = fill(div(n, k), k)
    for i in 1:rem(n, k)
        fold_sizes[i] += 1
    end

    folds = Vector{Vector{Int}}(undef, k)
    start_idx = 1
    for i in 1:k
        stop_idx = start_idx + fold_sizes[i] - 1
        folds[i] = idx[start_idx:stop_idx]
        start_idx = stop_idx + 1
    end
    return folds
end

# =========================
# Cross-validation function
# =========================
function run_kfold_sr(X, y, features, options; k=5, niterations=100, label="target")
    n = size(X, 1)
    folds = make_kfold_indices(n, k)

    rmse_train_all = Float64[]
    rmse_val_all   = Float64[]
    r2_train_all   = Float64[]
    r2_val_all     = Float64[]
    equations      = String[]

    for fold in 1:k
        println("\n==============================")
        println("$(label): Fold $fold / $k")
        println("==============================")

        val_idx = folds[fold]
        train_idx = vcat(folds[1:fold-1]..., folds[fold+1:end]...)

        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val   = X[val_idx, :]
        y_val   = y[val_idx]

        println("Training rows   = ", size(X_train, 1))
        println("Validation rows = ", size(X_val, 1))

        hof = equation_search(
            X_train', y_train;
            options = options,
            niterations = niterations,
            variable_names = features,
            parallelism = :multithreading,
        )

        dominating = calculate_pareto_frontier(hof)
        best_eq = dominating[end]

        eq_string = string_tree(best_eq.tree, options)
        push!(equations, eq_string)

        println("\nChosen equation:")
        println(eq_string)

        yhat_train, ok_train = eval_tree_array(best_eq.tree, X_train', options)
        yhat_val,   ok_val   = eval_tree_array(best_eq.tree, X_val', options)

        println("Train evaluation successful = ", ok_train)
        println("Val   evaluation successful = ", ok_val)

        if !ok_train || !ok_val
            println("Skipping fold $fold because evaluation failed.")
            continue
        end

        fold_rmse_train = rmse(y_train, yhat_train)
        fold_rmse_val   = rmse(y_val, yhat_val)
        fold_r2_train   = r2_score(y_train, yhat_train)
        fold_r2_val     = r2_score(y_val, yhat_val)

        push!(rmse_train_all, fold_rmse_train)
        push!(rmse_val_all, fold_rmse_val)
        push!(r2_train_all, fold_r2_train)
        push!(r2_val_all, fold_r2_val)

        println("\nFold metrics:")
        println("Train RMSE = ", fold_rmse_train)
        println("Val   RMSE = ", fold_rmse_val)
        println("Train R2   = ", fold_r2_train)
        println("Val   R2   = ", fold_r2_val)
    end

    return (
        rmse_train = rmse_train_all,
        rmse_val   = rmse_val_all,
        r2_train   = r2_train_all,
        r2_val     = r2_val_all,
        equations  = equations,
    )
end

# =========================
# Run 5-fold CV for P_bubble
# =========================
println("\n══════════════════════════════")
println("Running 5-fold CV for P_bubble")
println("══════════════════════════════")

results_bubble = run_kfold_sr(
    X, y_bubble, features, options;
    k = 5,
    niterations = 100,
    label = "P_bubble",
)

# =========================
# Run 5-fold CV for P_dew
# =========================
println("\n══════════════════════════════")
println("Running 5-fold CV for P_dew")
println("══════════════════════════════")

results_dew = run_kfold_sr(
    X, y_dew, features, options;
    k = 5,
    niterations = 100,
    label = "P_dew",
)

# =========================
# Summary
# =========================
println("\n===================================")
println("5-fold CV summary: P_bubble")
println("===================================")
println("Mean train RMSE = ", mean(results_bubble.rmse_train))
println("Mean val   RMSE = ", mean(results_bubble.rmse_val))
println("Mean train R2   = ", mean(results_bubble.r2_train))
println("Mean val   R2   = ", mean(results_bubble.r2_val))

println("\nFold-wise validation RMSE = ", results_bubble.rmse_val)
println("Fold-wise validation R2   = ", results_bubble.r2_val)

println("\n===================================")
println("5-fold CV summary: P_dew")
println("===================================")
println("Mean train RMSE = ", mean(results_dew.rmse_train))
println("Mean val   RMSE = ", mean(results_dew.rmse_val))
println("Mean train R2   = ", mean(results_dew.r2_train))
println("Mean val   R2   = ", mean(results_dew.r2_val))

println("\nFold-wise validation RMSE = ", results_dew.rmse_val)
println("Fold-wise validation R2   = ", results_dew.r2_val)

# =========================
# Save results
# =========================
serialize("outputs/kfold_results_bubble.jls", results_bubble)
serialize("outputs/kfold_results_dew.jls", results_dew)