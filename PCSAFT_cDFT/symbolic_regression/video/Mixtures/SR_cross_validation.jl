# ============================================================
#  5-fold cross-validation of the symbolic-regression feature sets
#  V1 (physics-informed, 12 features) and V2 (composition-only, 5)
#
#  Mirrors the CV protocol used by the GPR / SVGP residual models in
#  ../../ (RESIDUAL/GPR, RESIDUAL/SVGP):
#    * 5-fold StratifiedKFold, shuffled, fixed seed
#    * strata = T-P decile bins (q = 10), as in GPR.ipynb cell 10
#    * per-fold R2 / RMSE / MAE, reported as mean +/- 2*std
#
#  Because each SR fold refits the whole equation search, the CV score
#  measures the *generalisation of the feature set + search* (analogous to
#  refitting the GPR kernel each fold), not of a single fixed equation.
#  Within each fold an inner train/val split selects the best equation by
#  validation RMSE (same rule as select_best_equation in the production
#  Mixture.jl / Mixture_V2.jl runs).
#
#  Usage:
#     export JULIA_NUM_THREADS=<n>
#     julia SR_cross_validation.jl          # both versions
#     julia SR_cross_validation.jl V1       # only V1
#     julia SR_cross_validation.jl V2       # only V2
# ============================================================

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())

using CSV
using DataFrames
using Random
using Statistics
using Serialization
using SymbolicRegression
using JSON3
using Printf

cd(@__DIR__)
include("sr_utils.jl")

const OUTDIR = "SR_CV_OUTPUTS"
mkpath(OUTDIR)

mae(y, yhat) = mean(abs.(y .- yhat))

# ============================================================
# CV configuration (kept aligned with the GPR / SVGP residual runs)
# ============================================================
const CV_FOLDS       = 5
const CV_SEED        = 4555525        # same SEED used by GPR.ipynb / SVGP
const STRAT_BINS     = 10             # T-P quantile bins for stratification
const INNER_VAL_FRAC = 0.15           # held out *inside* each fold for eq. selection
# matches the production runs; override with e.g. SR_CV_NITERATIONS=20 for a quick test
const NITERATIONS    = parse(Int, get(ENV, "SR_CV_NITERATIONS", "200"))

# ============================================================
# Data loading + feature engineering (identical to Mixture.jl / Mixture_V2.jl)
# ============================================================
function load_dataframe()
    df = CSV.read("../../CombinedDatasetSEC_A4.csv", DataFrame; normalizenames=true)
    df[!, :gamma_base] = df[!, :gamma_wsd_UC]
    df[!, :gamma_cDFT] = df[!, :gamma_wsd_UC] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]
    df[!, :eps_base]   = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

    x_imp = [c for c in names(df) if startswith(c, "x_") && c != "x_carbon_dioxide"]
    y_imp = [c for c in names(df) if startswith(c, "y_") && c != "y_carbon_dioxide"]
    z_imp = [c for c in names(df) if startswith(c, "z_") && c != "z_carbon_dioxide"]

    df[!, :x_total_impurity]     = reduce(.+, [df[!, c] for c in x_imp])
    df[!, :y_total_impurity]     = reduce(.+, [df[!, c] for c in y_imp])
    df[!, :z_total_impurity]     = reduce(.+, [df[!, c] for c in z_imp])
    df[!, :delta_total_impurity] = df[!, :x_total_impurity] .- df[!, :y_total_impurity]

    eps = 1e-12
    df[!, :logK_total_impurity] = log.((df[!, :y_total_impurity] .+ eps) ./
                                       (df[!, :x_total_impurity] .+ eps))

    # Physics-informed (V1-only) features
    df[!, :Tr]               = df[!, :T] ./ df[!, :Tc_carbon_dioxide]
    df[!, :oneMinusTr]       = 1.0 .- df[!, :Tr]
    df[!, :Pr]               = df[!, :P] ./ df[!, :Pc_carbon_dioxide]
    df[!, :P_over_Psat0]     = df[!, :P] ./ df[!, :Psat0_carbon_dioxide]
    df[!, :delta_rho]        = df[!, :rho_l] .- df[!, :rho_v]
    df[!, :delta_rho_norm]   = (df[!, :rho_l] .- df[!, :rho_v]) ./
                               ((df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide]) .+ eps)
    df[!, :gamma_base_ratio] = df[!, :gamma_wsd_UC] ./ (df[!, :gamma0_carbon_dioxide] .+ eps)
    return df
end

const FEATURES_V1 = [
    :z_total_impurity, :x_total_impurity, :y_total_impurity,
    :delta_total_impurity, :logK_total_impurity,
    :Tr, :oneMinusTr, :Pr, :P_over_Psat0,
    :delta_rho, :delta_rho_norm, :gamma_base_ratio,
]
const FEATURES_V2 = [
    :z_total_impurity, :x_total_impurity, :y_total_impurity,
    :delta_total_impurity, :logK_total_impurity,
]

options_v1() = Options(
    binary_operators        = (+, -, *, safepow, safe_div),
    unary_operators         = (abs, safe_sqrt, safe_log),
    populations             = 120,
    maxsize                 = 24,
    parsimony               = 0.003f0,
    complexity_of_operators = [safe_div => 2, safepow => 3, abs => 2, safe_sqrt => 2, safe_log => 2],
    complexity_of_constants = 2,
    batching                = true,
    batch_size              = 1000,
)
options_v2() = Options(
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
# Stratified k-fold (replicates sklearn StratifiedKFold(shuffle=True))
# ============================================================
function decile_bins(x::AbstractVector, nbins::Int)
    edges = quantile(x, range(0, 1; length = nbins + 1))
    edges = unique(edges)                         # duplicates="drop"
    bins  = Vector{Int}(undef, length(x))
    @inbounds for i in eachindex(x)
        b = searchsortedlast(edges, x[i])
        bins[i] = clamp(b, 1, length(edges) - 1)
    end
    return bins
end

function stratified_kfold(strata::AbstractVector, k::Int, rng)
    folds = [Int[] for _ in 1:k]
    for s in unique(strata)
        idxs = findall(==(s), strata)
        shuffle!(rng, idxs)
        for (j, idx) in enumerate(idxs)
            push!(folds[mod1(j, k)], idx)
        end
    end
    return folds
end

# ============================================================
# Cross-validate one feature set
# ============================================================
function cross_validate(df::DataFrame, version::String, features::Vector{Symbol}, options)
    # Per-version filtering (V1 drops more rows because it has more features)
    cols = vcat(features, [:eps_base, :gamma_base, :T, :P])
    d = copy(df)
    dropmissing!(d, cols)
    filter!(row -> all(isfinite(row[c]) for c in cols), d)

    X          = Matrix{Float64}(d[:, features])
    y          = Vector{Float64}(d[:, :eps_base])
    gamma_base = Vector{Float64}(d[:, :gamma_base])
    Tv         = Vector{Float64}(d[:, :T])
    Pv         = Vector{Float64}(d[:, :P])
    n          = size(X, 1)

    # T-P decile strata, then stratified k folds
    strata = string.(decile_bins(Tv, STRAT_BINS)) .* "_" .* string.(decile_bins(Pv, STRAT_BINS))
    rng    = MersenneTwister(CV_SEED)
    folds  = stratified_kfold(strata, CV_FOLDS, rng)

    println("\n" * "="^60)
    println("Cross-validating $version  (n = $n, features = $(length(features)))")
    println("="^60)

    # Per-fold accumulators
    eps_r2   = Float64[]; eps_rmse  = Float64[]; eps_mae  = Float64[]
    g_r2     = Float64[]; g_rmse    = Float64[]; g_mae    = Float64[]
    complexities = Int[]

    for kf in 1:CV_FOLDS
        test_idx  = folds[kf]
        train_idx = setdiff(1:n, test_idx)

        # Inner train/val split (selection only — never touches the test fold)
        irng     = MersenneTwister(CV_SEED + kf)
        perm     = shuffle(irng, train_idx)
        n_val    = round(Int, INNER_VAL_FRAC * length(perm))
        val_idx  = perm[1:n_val]
        fit_idx  = perm[n_val+1:end]

        X_fit, y_fit = X[fit_idx, :], y[fit_idx]
        X_val, y_val = X[val_idx, :], y[val_idx]
        X_te,  y_te  = X[test_idx, :], y[test_idx]
        gb_te        = gamma_base[test_idx]

        println("\n── $version fold $kf/$CV_FOLDS ──  fit=$(length(fit_idx))  val=$(length(val_idx))  test=$(length(test_idx))")

        hof = equation_search(
            X_fit', y_fit;
            options        = options,
            niterations    = NITERATIONS,
            variable_names = string.(features),
            parallelism    = :multithreading,
        )

        dominating = calculate_pareto_frontier(hof)
        best_idx, _ = select_best_equation(dominating, X_val, y_val, options)
        best_eq    = dominating[best_idx]
        push!(complexities, compute_complexity(best_eq, options))

        yhat_te, ok = eval_tree_array(best_eq.tree, X_te', options)
        if !ok
            @warn "fold $kf: evaluation failed; skipping fold"
            continue
        end

        # eps_base metrics
        push!(eps_r2,   r2_score(y_te, yhat_te))
        push!(eps_rmse, rmse(y_te, yhat_te))
        push!(eps_mae,  mae(y_te, yhat_te))

        # Reconstructed gamma_cDFT (mN/m) — comparable to GPR/SVGP
        g_actual = gb_te .* (1.0 .+ y_te)
        g_pred   = gb_te .* (1.0 .+ yhat_te)
        push!(g_r2,   r2_score(g_actual, g_pred))
        push!(g_rmse, rmse(g_actual, g_pred))
        push!(g_mae,  mae(g_actual, g_pred))

        println("   fold $kf:  eps R²=$(round(eps_r2[end],sigdigits=4))  " *
                "eps RMSE=$(round(eps_rmse[end],sigdigits=4))  " *
                "γ RMSE=$(round(g_rmse[end],sigdigits=4)) mN/m  " *
                "(complexity $(complexities[end]))")
    end

    summary = Dict(
        "version"          => version,
        "n_samples"        => n,
        "n_features"       => length(features),
        "cv_folds"         => CV_FOLDS,
        "cv_seed"          => CV_SEED,
        "niterations"      => NITERATIONS,
        "selected_complexity_per_fold" => complexities,
        # eps_base (dimensionless)
        "eps_r2_folds"     => eps_r2,
        "eps_rmse_folds"   => eps_rmse,
        "eps_mae_folds"    => eps_mae,
        "eps_r2_mean"      => mean(eps_r2),   "eps_r2_std"   => std(eps_r2),
        "eps_rmse_mean"    => mean(eps_rmse), "eps_rmse_std" => std(eps_rmse),
        "eps_mae_mean"     => mean(eps_mae),  "eps_mae_std"  => std(eps_mae),
        # reconstructed gamma_cDFT (mN/m)
        "gamma_r2_folds"   => g_r2,
        "gamma_rmse_folds" => g_rmse,
        "gamma_mae_folds"  => g_mae,
        "gamma_r2_mean"    => mean(g_r2),   "gamma_r2_std"   => std(g_r2),
        "gamma_rmse_mean"  => mean(g_rmse), "gamma_rmse_std" => std(g_rmse),
        "gamma_mae_mean"   => mean(g_mae),  "gamma_mae_std"  => std(g_mae),
    )

    println("\n── $version  5-fold CV summary (mean ± 2·std) ──")
    @printf("eps_base   R²   = %.4f ± %.4f\n", mean(eps_r2),   2std(eps_r2))
    @printf("eps_base   RMSE = %.4f ± %.4f\n", mean(eps_rmse), 2std(eps_rmse))
    @printf("eps_base   MAE  = %.4f ± %.4f\n", mean(eps_mae),  2std(eps_mae))
    @printf("γ_cDFT     R²   = %.5f ± %.5f\n", mean(g_r2),     2std(g_r2))
    @printf("γ_cDFT     RMSE = %.4f ± %.4f mN/m\n", mean(g_rmse), 2std(g_rmse))
    @printf("γ_cDFT     MAE  = %.4f ± %.4f mN/m\n", mean(g_mae),  2std(g_mae))
    println("complexity per fold = ", complexities)

    return summary
end

# ============================================================
# Main
# ============================================================
function main()
    which = isempty(ARGS) ? ["V1", "V2"] : ARGS
    df = load_dataframe()
    results = Dict{String,Any}()

    if "V1" in which
        results["V1"] = cross_validate(df, "V1", FEATURES_V1, options_v1())
    end
    if "V2" in which
        results["V2"] = cross_validate(df, "V2", FEATURES_V2, options_v2())
    end

    # Version-specific filename so parallel single-version runs don't clobber
    # each other (e.g. "V1" -> SR_CV_metrics_V1.json). Finalisation reads any
    # SR_CV_metrics_*.json present.
    tag     = join(which, "_")
    outpath = joinpath(OUTDIR, "SR_CV_metrics_$(tag).json")
    open(outpath, "w") do io
        JSON3.pretty(io, results)
    end
    println("\nSaved CV metrics JSON: ", outpath)
end

main()
