
using CSV
using DataFrames
using Statistics
using Random
using SymbolicRegression
using SpecialFunctions
using Plots
using Base.Threads

 

include("sr_utils.jl")

 

rmse(ytrue, ypred) = sqrt(mean((ytrue .- ypred).^2))
r2(ytrue, ypred) = 1 - sum((ytrue .- ypred).^2) / sum((ytrue .- mean(ytrue)).^2)

 

function finite_metrics(ytrue, ypred)
    mask = isfinite.(ytrue) .& isfinite.(ypred)
    if count(mask) == 0
        return (rmse=Inf, r2=-Inf, n=0)
    end
    yt = ytrue[mask]
    yp = ypred[mask]
    return (rmse=rmse(yt, yp), r2=r2(yt, yp), n=length(yt))
end

 

function save_parity_plot(
    ytrue_train, ypred_train,
    ytrue_val, ypred_val,
    ytrue_test, ypred_test,
    eq_idx, outpath;
    target_name="N_CO2"
)
    mask_train = isfinite.(ytrue_train) .& isfinite.(ypred_train)
    mask_val   = isfinite.(ytrue_val)   .& isfinite.(ypred_val)
    mask_test  = isfinite.(ytrue_test)  .& isfinite.(ypred_test)

 

    yt_train = ytrue_train[mask_train]
    yp_train = ypred_train[mask_train]
    yt_val   = ytrue_val[mask_val]
    yp_val   = ypred_val[mask_val]
    yt_test  = ytrue_test[mask_test]
    yp_test  = ypred_test[mask_test]

 

    allx = vcat(yt_train, yt_val, yt_test)
    ally = vcat(yp_train, yp_val, yp_test)

 

    xmin = minimum(allx)
    xmax = maximum(allx)
    ymin = minimum(ally)
    ymax = maximum(ally)

 

    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

 

    mtrain = finite_metrics(ytrue_train, ypred_train)
    mval   = finite_metrics(ytrue_val, ypred_val)
    mtest  = finite_metrics(ytrue_test, ypred_test)

 

    p = scatter(
        yt_train, yp_train;
        xlabel = "Actual $target_name",
        ylabel = "Predicted $target_name",
        label = "Train",
        markersize = 4,
        title = "Parity Plot for $target_name (Eq. $eq_idx)",
        legend = :bottomright,
    )

 

    scatter!(p, yt_val, yp_val; label = "Validation", markersize = 3)
    scatter!(p, yt_test, yp_test; label = "Test", markersize = 3)

 

    plot!(
        p,
        [lo, hi], [lo, hi];
        label = "y = x",
        lw = 2,
        color = :black,
        line = :dash
    )

 

    x_text = xmin + 0.1 * (xmax - xmin)
    y_top  = ymin + 0.96 * (ymax - ymin)
    dy     = 0.075 * (ymax - ymin)

 

    annotate!(p, x_text, y_top - 0 * dy, text("Eq. index = $eq_idx", 9, :left))
    annotate!(p, x_text, y_top - 1 * dy, text("Train RMSE = $(round(mtrain.rmse, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 2 * dy, text("Train R² = $(round(mtrain.r2, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 3 * dy, text("Val RMSE = $(round(mval.rmse, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 4 * dy, text("Val R² = $(round(mval.r2, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 5 * dy, text("Test RMSE = $(round(mtest.rmse, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 6 * dy, text("Test R² = $(round(mtest.r2, sigdigits=4))", 9, :left))

 

    savefig(p, outpath)
end

 

function main(subdir="test_1")
    outdir = joinpath("outputs", subdir)
    mkpath(outdir)

 

    println("Julia threads available = ", Threads.nthreads())
    println("Julia thread id         = ", Threads.threadid())

 

    # Backup a copy of this script into the output folder
    script_path = abspath(PROGRAM_FILE)
    script_base = splitext(basename(script_path))[1]
    #timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    backup_path = joinpath(outdir, "$(script_base)_backup.txt")
    cp(script_path, backup_path; force=true)

 

    println("Backed up script to: ", backup_path)

 

    # --------------------------------------------------
    # 1) Load data
    # --------------------------------------------------
    df = CSV.read("combined_results_ND_case1_303.csv", DataFrame)

 

    # --------------------------------------------------
    # 2) Rename awkward columns
    # --------------------------------------------------
    rename!(df, Dict(
        Symbol("N_CO2'")     => :N_CO2,
        Symbol("D_H2'")      => :D_H2,
        Symbol("C_H2'")      => :C_H2,
        Symbol("C_H'")       => :C_H,
        Symbol("C_H2PO4'")   => :C_H2PO4,
        Symbol("K_CO2'")     => :K_CO2,
        Symbol("K_H2'")      => :K_H2,
        Symbol("Keq_CO2'")   => :Keq_CO2,
        Symbol("kf_CO2'")    => :kf_CO2,
        Symbol("Keq_H2PO4'") => :Keq_H2PO4,
        Symbol("kf_H2PO4'")  => :kf_H2PO4,
        Symbol("C_HCO3'")  => :C_HCO3,
    ))

 

    # --------------------------------------------------
    # 3) Feature engineering
    # --------------------------------------------------
    target = :N_CO2

 

    df.Da_H2 = df.Da_act ./ (df.C_H2 .* df.D_H2)
    df.Da_CO2 = df.Da_act

 

    df.H2Hyper = df.C_H2 ./ (df.C_H2 .+ df.K_H2)
    df.CO2Hyper = 1 ./ (1 .+ df.K_CO2)

    #df.H = 10 .^ (-df.pH) ./ 10 .^(-5.8)
    df.H = df.C_H
    df.CO2Buffer_model = df.kf_CO2 ./ (1 .+ (df.H ./ df.Keq_CO2))
    #df.CO2Buffer_model = df.kf_CO2 .* (1 .- (df.H .* df.C_HCO3) ./ df.Keq_CO2)
    df.PBRBuffer_model = df.kf_H2PO4 ./ (1 .+ (df.H ./ df.Keq_H2PO4))
    df.Da_ratio = df.Da_ec ./ df.Da_CO2

 

    df.CO2_driving = df.CO2Hyper .* df.Da_CO2 .* df.H2Hyper

 

    df.H2_comp =
    df.Da_ec ./ (
        df.Da_ec .+
        4 .* df.Da_act .+
        4 .* df.Da_but .+
        4 .* df.Da_cap
    )

 

    df.CO2_buffered_drive = df.CO2Buffer_model .* df.CO2_driving
    df.log_Da_CO2 = [x > 0 ? log(x) : NaN for x in df.Da_CO2]
    df.log_kf_CO2 = [x > 0 ? log(x) : NaN for x in df.kf_CO2]
    df.log_Keq_CO2 = [x > 0 ? log(x) : NaN for x in df.Keq_CO2]

 

    df.H2kernel = [c > -1 ? c - log1p(c) : NaN for c in df.C_H2]
    df.H2flux   = [c > -1 ? sqrt(max(0, 2 * (c - log1p(c)))) : NaN for c in df.C_H2]
    df.kf_log   = [k > 0 ? log(k) : NaN for k in df.kf_H2PO4]

 

    feature_cols = [
        :CO2Hyper,
        :Da_CO2,
        :Da_ec,
        :kf_CO2,
        :Keq_CO2,
        :H2Hyper,
        :H2_comp,
        :D_H2,
        #:H,
        #:pH,
        :PBRBuffer_model,
        :CO2Buffer_model,
        #:Da_ratio,
        :CO2_driving,
        :CO2_buffered_drive,
        :log_Da_CO2,
        :log_kf_CO2,
        :log_Keq_CO2,
    ]

 

    # --------------------------------------------------
    # 4) Filter finite rows
    # --------------------------------------------------
    valid_mask = trues(nrow(df))
    for col in feature_cols
        valid_mask .&= isfinite.(df[!, col])
    end
    valid_mask .&= isfinite.(df[!, target])

 

    df = df[valid_mask, :]
    println("Rows kept after filtering = ", nrow(df))

 

    println("\nFeature standard deviations:")
    for col in feature_cols
        println(rpad(string(col), 20), " => ", std(df[!, col]))
    end

 

    # --------------------------------------------------
    # 5) Build X and y
    # --------------------------------------------------
    Xdf = select(df, feature_cols)
    X = permutedims(Matrix{Float32}(Xdf))
    y = Vector{Float32}(df[:, target])

 

    # --------------------------------------------------
    # 6) Train / validation / test split
    # --------------------------------------------------
    Random.seed!(1000)

 

    n = length(y)
    idx = collect(1:n)
    shuffle!(idx)

 

    ntrain = round(Int, 0.70 * n)
    nval   = round(Int, 0.15 * n)

 

    train_idx = idx[1:ntrain]
    val_idx   = idx[ntrain+1 : ntrain+nval]
    test_idx  = idx[ntrain+nval+1 : end]

 

    Xtrain = X[:, train_idx]
    ytrain = y[train_idx]

 

    Xval = X[:, val_idx]
    yval = y[val_idx]

 

    Xtest = X[:, test_idx]
    ytest = y[test_idx]

 

    println("\nSplit sizes:")
    println("Train = ", length(ytrain))
    println("Val   = ", length(yval))
    println("Test  = ", length(ytest))

 

    # --------------------------------------------------
    # 7) Symbolic regression options
    # --------------------------------------------------
    safepow(x::T, a::T) where {T<:Real} = x > zero(T) ? x^a : T(NaN)
    safesqrt(x::T) where {T<:Real} = x >= zero(T) ? sqrt(x) : T(NaN)
    safecbrt(x::T) where {T<:Real} = x >= zero(T) ? (x)^(1/3) : T(NaN)
    safe_erfc(x::T) where {T<:Real} = begin
        if !isfinite(x)
            return T(NaN)
        end
        y = erfc(x)
        isfinite(y) ? T(y) : T(NaN)
    end
    # try erfc
    options = Options(
        binary_operators = [+, -, *, /, safepow],
        unary_operators  = [exp, log, tanh, safesqrt],
        constraints = [

 

        safepow => (-1, 0),

 

    ],

 

    nested_constraints = [

 

        safepow => [safepow => 0],

 

    ],
        complexity_of_operators = Dict(tanh => 1.5,safepow =>0.1, safesqrt =>1, log =>3),
        maxsize = 30,
        populations = max(Threads.nthreads(), 36),
        population_size = 160,
        batching = true,
        batch_size = 256,
        parsimony = 6f0,
    )

 

    hall_of_fame = equation_search(
        Xtrain,
        ytrain;
        niterations = 5500,
        options = options,
        parallelism = :multithreading,
        variable_names = string.(feature_cols)
    )

 

    dominating = calculate_pareto_frontier(hall_of_fame)

 

    println("\n==============================")
    println("Pareto-front equations")
    println("==============================\n")

 

    for (i, eq) in enumerate(dominating)
        println("Equation $i")
        println(eq)
        println("Tree: ", eq.tree)
        println()
    end

 

    # --------------------------------------------------
    # 8) Evaluate equations on train / val / test
    #    Parallel compute, serial collect
    # --------------------------------------------------
    neqs = length(dominating)

 

    metrics_buffer = Vector{NamedTuple}(undef, neqs)

 

    println("\n==============================")
    println("Train / Validation / Test metrics")
    println("==============================\n")

 

    Threads.@threads for i in 1:neqs
        member = dominating[i]
        tree = member.tree

 

        yhat_train = tree(Xtrain)
        yhat_val   = tree(Xval)
        yhat_test  = tree(Xtest)

 

        mtrain = finite_metrics(ytrain, yhat_train)
        mval   = finite_metrics(yval, yhat_val)
        mtest  = finite_metrics(ytest, yhat_test)

 

        metrics_buffer[i] = (
            Equation = i,
            Train_RMSE = mtrain.rmse,
            Val_RMSE = mval.rmse,
            Test_RMSE = mtest.rmse,
            Train_R2 = mtrain.r2,
            Val_R2 = mval.r2,
            Test_R2 = mtest.r2,
            NTrain = mtrain.n,
            NVal = mval.n,
            NTest = mtest.n,
            Expression = string(tree)
        )
    end

 

    results = DataFrame(metrics_buffer)

 

    for row in eachrow(results)
        println("Equation $(row.Equation)")
        println("  expression : ", row.Expression)
        println("  train RMSE : ", row.Train_RMSE)
        println("  val   RMSE : ", row.Val_RMSE)
        println("  test  RMSE : ", row.Test_RMSE)
        println("  train R²   : ", row.Train_R2)
        println("  val   R²   : ", row.Val_R2)
        println("  test  R²   : ", row.Test_R2)
        println()
    end

 

    sort!(results, :Val_RMSE)

 

    best_i = results.Equation[1]
    best_val_rmse = results.Val_RMSE[1]

 

    println("Best equation by validation RMSE = Equation $best_i")
    println("Expression: ", dominating[best_i].tree)
    println("Best validation RMSE = ", best_val_rmse)

 

    CSV.write(joinpath(outdir, "NCO2_equation_ranking.csv"), results)

 

    # --------------------------------------------------
    # 9) Best equation predictions
    # --------------------------------------------------
    best_tree = dominating[best_i].tree

 

    yhat_train_best = best_tree(Xtrain)
    yhat_val_best   = best_tree(Xval)
    yhat_test_best  = best_tree(Xtest)

 

    CSV.write(
        joinpath(outdir, "NCO2_train_predictions_best.csv"),
        DataFrame(Actual = ytrain, Predicted = yhat_train_best)
    )

 

    CSV.write(
        joinpath(outdir, "NCO2_validation_predictions_best.csv"),
        DataFrame(Actual = yval, Predicted = yhat_val_best)
    )

 

    CSV.write(
        joinpath(outdir, "NCO2_test_predictions_best.csv"),
        DataFrame(Actual = ytest, Predicted = yhat_test_best)
    )

 

    save_parity_plot(
        ytrain, yhat_train_best,
        yval, yhat_val_best,
        ytest, yhat_test_best,
        best_i,
        joinpath(outdir, "NCO2_parity_best.png");
        target_name = "N_CO2"
    )

 

    # --------------------------------------------------
    # 10) Save parity plots and predictions for top equations
    #    Keep serial because plotting/file I/O is not reliably thread-safe
    # --------------------------------------------------
    top_k = min(5, nrow(results))

 

    for row in eachrow(results[1:top_k, :])
        eq_idx = row.Equation
        tree = dominating[eq_idx].tree

 

        yhat_train = tree(Xtrain)
        yhat_val   = tree(Xval)
        yhat_test  = tree(Xtest)

 

        CSV.write(
            joinpath(outdir, "NCO2_predictions_eq$(eq_idx).csv"),
            DataFrame(
                Split = vcat(
                    fill("Train", length(ytrain)),
                    fill("Validation", length(yval)),
                    fill("Test", length(ytest))
                ),
                Actual = vcat(ytrain, yval, ytest),
                Predicted = vcat(yhat_train, yhat_val, yhat_test)
            )
        )

 

        save_parity_plot(
            ytrain, yhat_train,
            yval, yhat_val,
            ytest, yhat_test,
            eq_idx,
            joinpath(outdir, "NCO2_parity_eq$(eq_idx).png");
            target_name = "N_CO2"
        )
    end

 

    # --------------------------------------------------
    # 11) Residual diagnostics for best equation
    # --------------------------------------------------
    resid_test = ytest .- yhat_test_best
    resid_df = DataFrame(
        Actual = ytest,
        Predicted = yhat_test_best,
        Residual = resid_test
    )
    CSV.write(joinpath(outdir, "NCO2_test_residuals_best.csv"), resid_df)

 

    test_df = df[test_idx, :]

 

    p1 = scatter(test_df.pH, resid_test, xlabel="PH", ylabel="Residual", title="Residual vs PH", markersize=3, label="")
    savefig(p1, joinpath(outdir, "Residual_vs_PH.png"))

 

    p2 = scatter(test_df.Da_CO2, resid_test, xlabel="Da_CO2", ylabel="Residual", title="Residual vs Da_CO2", markersize=3, label="")
    savefig(p2, joinpath(outdir, "Residual_vs_Da_CO2.png"))

 

    p3 = scatter(test_df.kf_CO2, resid_test, xlabel="kf_CO2", ylabel="Residual", title="Residual vs kf_CO2", markersize=3, label="")
    savefig(p3, joinpath(outdir, "Residual_vs_kf_CO2.png"))

 

    p4 = scatter(test_df.Keq_CO2, resid_test, xlabel="Keq_CO2", ylabel="Residual", title="Residual vs Keq_CO2", markersize=3, label="")
    savefig(p4, joinpath(outdir, "Residual_vs_Keq_CO2.png"))

 

    # --------------------------------------------------
    # 12) Save equations
    # --------------------------------------------------
    open(joinpath(outdir, "NCO2_symbolic_equations.txt"), "w") do io
        println(io, "Feature columns used:")
        println(io, feature_cols)
        println(io)

 

        println(io, "Best equation by validation RMSE = Equation $best_i")
        println(io, "Expression: ", dominating[best_i].tree)
        println(io)

 

        for (i, member) in enumerate(dominating)
            println(io, "Equation $i")
            println(io, member)
            println(io, "Tree: ", member.tree)
            println(io)
        end
    end

 

    # --------------------------------------------------
    # 13) Save a run summary
    # --------------------------------------------------
    best_train = finite_metrics(ytrain, yhat_train_best)
    best_val   = finite_metrics(yval, yhat_val_best)
    best_test  = finite_metrics(ytest, yhat_test_best)

 

    open(joinpath(outdir, "run_summary.txt"), "w") do io
        println(io, "Output directory: ", outdir)
        println(io, "Target: ", target)
        println(io, "Feature columns:")
        println(io, feature_cols)
        println(io)
        println(io, "Train size = ", length(ytrain))
        println(io, "Val size   = ", length(yval))
        println(io, "Test size  = ", length(ytest))
        println(io)
        println(io, "Julia threads = ", Threads.nthreads())
        println(io, "Best equation index (by validation RMSE) = ", best_i)
        println(io, "Best expression = ", dominating[best_i].tree)
        println(io)
        println(io, "Train RMSE = ", best_train.rmse)
        println(io, "Train R²   = ", best_train.r2)
        println(io, "Val RMSE   = ", best_val.rmse)
        println(io, "Val R²     = ", best_val.r2)
        println(io, "Test RMSE  = ", best_test.rmse)
        println(io, "Test R²    = ", best_test.r2)
    end

 

    println("\nSaved files in: ", outdir)
    println("  ", joinpath(outdir, "NCO2_equation_ranking.csv"))
    println("  ", joinpath(outdir, "NCO2_train_predictions_best.csv"))
    println("  ", joinpath(outdir, "NCO2_validation_predictions_best.csv"))
    println("  ", joinpath(outdir, "NCO2_test_predictions_best.csv"))
    println("  ", joinpath(outdir, "NCO2_parity_best.png"))
    println("  ", joinpath(outdir, "NCO2_symbolic_equations.txt"))
    println("  ", joinpath(outdir, "run_summary.txt"))
end

 

if length(ARGS) > 0
    main(ARGS[1])
else
    main("N_CO2_18-0d")
end

using CSV
using DataFrames
using Statistics
using Random
using SymbolicRegression
using Plots
using Base.Threads

 

include("sr_utils.jl")

 

rmse(ytrue, ypred) = sqrt(mean((ytrue .- ypred).^2))
r2(ytrue, ypred) = 1 - sum((ytrue .- ypred).^2) / sum((ytrue .- mean(ytrue)).^2)

 

function finite_metrics(ytrue, ypred)
    mask = isfinite.(ytrue) .& isfinite.(ypred)
    if count(mask) == 0
        return (rmse=Inf, r2=-Inf, n=0)
    end
    yt = ytrue[mask]
    yp = ypred[mask]
    return (rmse=rmse(yt, yp), r2=r2(yt, yp), n=length(yt))
end

 

function save_parity_plot(
    ytrue_train, ypred_train,
    ytrue_val, ypred_val,
    ytrue_test, ypred_test,
    eq_idx, outpath;
    target_name="NC2"
)
    mask_train = isfinite.(ytrue_train) .& isfinite.(ypred_train)
    mask_val   = isfinite.(ytrue_val)   .& isfinite.(ypred_val)
    mask_test  = isfinite.(ytrue_test)  .& isfinite.(ypred_test)

 

    yt_train = ytrue_train[mask_train]
    yp_train = ypred_train[mask_train]
    yt_val   = ytrue_val[mask_val]
    yp_val   = ypred_val[mask_val]
    yt_test  = ytrue_test[mask_test]
    yp_test  = ypred_test[mask_test]

 

    allx = vcat(yt_train, yt_val, yt_test)
    ally = vcat(yp_train, yp_val, yp_test)

 

    xmin = minimum(allx)
    xmax = maximum(allx)
    ymin = minimum(ally)
    ymax = maximum(ally)

 

    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

 

    mtrain = finite_metrics(ytrue_train, ypred_train)
    mval   = finite_metrics(ytrue_val, ypred_val)
    mtest  = finite_metrics(ytrue_test, ypred_test)

 

    p = scatter(
        yt_train, yp_train;
        xlabel = "Actual $target_name",
        ylabel = "Predicted $target_name",
        label = "Train",
        markersize = 4,
        title = "Parity Plot for $target_name (Eq. $eq_idx)",
        legend = :bottomright,
    )

 

    scatter!(p, yt_val, yp_val; label = "Validation", markersize = 3)
    scatter!(p, yt_test, yp_test; label = "Test", markersize = 3)

 

    plot!(
        p,
        [lo, hi], [lo, hi];
        label = "y = x",
        lw = 2,
        color = :black,
        line = :dash
    )

 

    x_text = xmin + 0.1 * (xmax - xmin)
    y_top  = ymin + 0.96 * (ymax - ymin)
    dy     = 0.075 * (ymax - ymin)

 

    annotate!(p, x_text, y_top - 0 * dy, text("Eq. index = $eq_idx", 9, :left))
    annotate!(p, x_text, y_top - 1 * dy, text("Train RMSE = $(round(mtrain.rmse, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 2 * dy, text("Train R² = $(round(mtrain.r2, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 3 * dy, text("Val RMSE = $(round(mval.rmse, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 4 * dy, text("Val R² = $(round(mval.r2, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 5 * dy, text("Test RMSE = $(round(mtest.rmse, sigdigits=4))", 9, :left))
    annotate!(p, x_text, y_top - 6 * dy, text("Test R² = $(round(mtest.r2, sigdigits=4))", 9, :left))

 

    savefig(p, outpath)
end

 

function main(subdir="test_1")
    outdir = joinpath("outputs", subdir)
    mkpath(outdir)

 

    println("Julia threads available = ", Threads.nthreads())
    println("Julia thread id         = ", Threads.threadid())

 

    # Backup a copy of this script into the output folder
    script_path = abspath(PROGRAM_FILE)
    script_base = splitext(basename(script_path))[1]
    #timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    backup_path = joinpath(outdir, "$(script_base)_backup.txt")
    cp(script_path, backup_path; force=true)

 

    println("Backed up script to: ", backup_path)

 

    # --------------------------------------------------
    # 1) Load data
    # --------------------------------------------------
    df = CSV.read("combined_results_ND_case1_303.csv", DataFrame)

 

    # --------------------------------------------------
    # 2) Rename awkward columns
    # --------------------------------------------------
    rename!(df, Dict(
        Symbol("N_CO2'")     => :N_CO2,
        Symbol("N_H2'")     => :N_H2,
        Symbol("N_C2'")     => :N_C2,
        Symbol("N_C4'")     => :N_C4,
        Symbol("N_C6'")     => :N_C6,
        Symbol("N_HCO3'")     => :N_HCO3,
        Symbol("D_H2'")      => :D_H2,
        Symbol("D_C2'")      => :D_C2,
        Symbol("D_C4'")      => :D_C4,
        Symbol("D_C6'")      => :D_C6,
        Symbol("C_H2'")      => :C_H2,
        Symbol("C_C2'")      => :C_C2,
        Symbol("C_C4'")      => :C_C4,
        Symbol("C_C6'")      => :C_C6,
        Symbol("C_H'")       => :C_H,
        Symbol("C_H2PO4'")   => :C_H2PO4,
        Symbol("C_HCO3'")   => :C_HCO3,
        Symbol("K_CO2'")     => :K_CO2,
        Symbol("K_H2'")      => :K_H2,
        Symbol("K_C2'")      => :K_C2,
        Symbol("K_C4'")      => :K_C4,
        Symbol("Keq_CO2'")   => :Keq_CO2,
        Symbol("kf_CO2'")    => :kf_CO2,
        Symbol("Keq_H2PO4'") => :Keq_H2PO4,
        Symbol("kf_H2PO4'")  => :kf_H2PO4
    ))

 

    # --------------------------------------------------
    # CUSTOM FUNCTIONS BASED ON MY PHYSICAL INTERPRETATION
    # --------------------------------------------------
    target = :N_C2

 

    # # damkohler number for acetogenesis
    # df.Da_H2 = 2 .* df.Da_act ./ (df.C_H2 .* df.D_H2)
    df.Da_CO2 = df.Da_act
    # # df.Da_act = 0.5 .* df.Da_act ./ (df.C_C2 .* df.D_C2)
    # # df.Da_act = 0.5 .* df.Da_act 
    # df.frac_to_CO2 = df.Da_CO2 ./ (df.Da_CO2 .+ df.Da_but .+ df.Da_cap)

 

 

    # # damkohler number of butyrate elongation
    # #df.Da_but_act = -2 .* df.Da_but ./ (df.C_C2 .* df.D_C2)
    #  df.Da_but_act = -2 .* df.Da_but 
    # df.Da_but_H2 = 2 .* df.Da_but ./ (df.C_H2 .* df.D_H2)
    # df.Da_but_but = df.Da_but ./ (df.C_C4 .* df.D_C4)

 

    # # damkohler number of caproate elongation
    # # df.Da_cap_act = -df.Da_cap ./ (df.C_C2 .* df.D_C2)
    # df.Da_cap_act = -df.Da_cap 
    # df.Da_cap_but = df.Da_cap ./ (df.C_C4 .* df.D_C4)
    # df.Da_cap_H2 = 2 .* df.Da_cap ./ (df.C_H2 .* df.D_H2)
    # df.Da_net_act = df.Da_act .+ df.Da_but_act .+ df.Da_cap_act
    # #Hyperbolic functions
    df.H2Hyper = df.C_H2 ./ (df.C_H2 .+ df.K_H2)
    df.CO2Hyper = 1 ./ (1 .+ df.K_CO2)
    df.C2Hyper = df.C_C2 ./ (df.C_C2 .+ df.K_C2)
    df.C4Hyper = df.C_C4 ./ (df.C_C4 .+ df.K_C4)
    # df.H = df.C_H

    # df.CO2Buffer_model = df.kf_CO2 .* (1 .- (df.H .* df.C_HCO3) ./ df.Keq_CO2)
    # df.PBRBuffer_model = df.kf_H2PO4 ./ (1 .+ (df.H ./ df.Keq_H2PO4))

    # df.Da_ratio = df.Da_ec ./ df.Da_CO2
    df.Da_ec_sat = df.Da_ec ./ (1 .+ df.Da_ec)

 

    # df.CO2_driving = df.CO2Hyper .* df.Da_CO2 .* df.H2Hyper 
    # df.Act_driving_act = df.Da_act .*  df.CO2Hyper .* df.H2Hyper
    # df.Act_driving_but = df.Da_but_act .* df.C2Hyper .* df.H2Hyper
    # df.Act_driving_cap = df.Da_cap_act .* df.C2Hyper .* df.C4Hyper .* df.H2Hyper
    # df.Da_ec_Act = df.Da_ec ./ (df.C_C2 .* df.D_C2)

 

 

    df.H2_comp =
    df.Da_ec ./ (
        df.Da_ec .+
        4 .* df.Da_act .+
        4 .* df.Da_but .+
        4 .* df.Da_cap
    )

 

    df.CO2_drive = 0.5 .* df.Da_act .* df.CO2Hyper .* df.H2Hyper
    df.BUT_drive = 2 .* df.Da_but .* df.C2Hyper .* df.H2Hyper
    df.CAP_drive = df.Da_cap .* df.C2Hyper .* df.C4Hyper .* df.H2Hyper  
    df.Net = df.CO2_drive .- df.BUT_drive .- df.CAP_drive

 

    df.C2_comp_max = df.Da_act ./ (df.Da_act .+ 4 .* df.Da_but .+ 2 .* df.Da_cap)
    df.C2_comp  = df.CO2_drive ./ (df.CO2_drive .+ df.BUT_drive .+ df.CAP_drive)

    df.C2_supply_ratio  = df.CO2_drive ./ (df.BUT_drive .+ df.CAP_drive)
    df.regime = tanh.(df.C2_supply_ratio)
    # df.total_comp = df.CO2_drive .+ df.BUT_drive .+ df.CAP_drive
    # df.CO2_buffered_drive = df.CO2Buffer_model .* df.CO2_driving
    # df.log_Da_CO2 = [x > 0 ? log(x) : NaN for x in df.Da_CO2]
    # df.log_kf_CO2 = [x > 0 ? log(x) : NaN for x in df.kf_CO2]
    # df.log_Keq_CO2 = [x > 0 ? log(x) : NaN for x in df.Keq_CO2]
    # df.log_kf_H2PO4 = [x > 0 ? log(x) : NaN for x in df.kf_H2PO4]
    # df.log_Keq_H2PO4 = [x > 0 ? log(x) : NaN for x in df.Keq_H2PO4]
    # df.log10_H = [x > 0 ? log10(x) : NaN for x in df.H]

 

    # df.H2kernel = [c > -1 ? c - log1p(c) : NaN for c in df.C_H2]
    # df.H2flux   = [c > -1 ? sqrt(max(0, 2 * (c - log1p(c)))) : NaN for c in df.C_H2]
    # df.kf_log   = [k > 0 ? log(k) : NaN for k in df.kf_H2PO4]

 

    # feature_cols = [
    #     :Da_act,
    #     :frac_to_CO2,
    #     # :Da_net_act,
    #     :Da_but_act,
    #     :Da_cap_act,
    #     :Da_ec,
    #     # :Act_driving_act,
    #     # :Act_driving_but,
    #     # :Act_driving_cap,
    #     :Da_ec_sat,
    #     :H2Hyper,
    #     :C2Hyper,
    #     :C4Hyper,
    #     :CO2Hyper,
    #     # :H
    #     # :D_C2,
    #     # :D_C4,
    #     # :C_C2,
    #     # :C_C4,
    #     # :C_H2,
    #     # :K_C2,
    #     # :K_H2,
    #     # :K_C4,
    # ]
    feature_cols = [
    # :Da_ec_sat,
    :Net,
    :Da_ec,
    :C2_comp,
    :C2_supply_ratio,
    :C2Hyper,
    # :CO2Hyper,
    # :Da_ec_sat,
    :H2Hyper,
    :regime,
    # :Da_act,
    # :Da_but,
    # :Da_cap,
    :H2_comp,
    # :D_C2,
    # :D_C4,
    # :D_C6,
    # :D_H2,
    :CO2_drive,
    :BUT_drive,
    :CAP_drive,
]

 

    # --------------------------------------------------
    # 4) Filter finite rows
    # --------------------------------------------------
    valid_mask = trues(nrow(df))
    for col in feature_cols
        valid_mask .&= isfinite.(df[!, col])
    end
    valid_mask .&= isfinite.(df[!, target])

 

    df = df[valid_mask, :]
    println("Rows kept after filtering = ", nrow(df))

 

    println("\nFeature standard deviations:")
    for col in feature_cols
        println(rpad(string(col), 20), " => ", std(df[!, col]))
    end

 

    # --------------------------------------------------
    # 5) Build X and y
    # --------------------------------------------------
    Xdf = select(df, feature_cols)
    X = permutedims(Matrix{Float32}(Xdf))
    y = Vector{Float32}(df[:, target])

 

    # --------------------------------------------------
    # 6) Train / validation / test split
    # --------------------------------------------------
    Random.seed!(1000)

 

    n = length(y)
    idx = collect(1:n)
    shuffle!(idx)

 

    ntrain = round(Int, 0.70 * n)
    nval   = round(Int, 0.15 * n)

 

    train_idx = idx[1:ntrain]
    val_idx   = idx[ntrain+1 : ntrain+nval]
    test_idx  = idx[ntrain+nval+1 : end]

 

    Xtrain = X[:, train_idx]
    ytrain = y[train_idx]

 

    Xval = X[:, val_idx]
    yval = y[val_idx]

 

    Xtest = X[:, test_idx]
    ytest = y[test_idx]

 

    println("\nSplit sizes:")
    println("Train = ", length(ytrain))
    println("Val   = ", length(yval))
    println("Test  = ", length(ytest))

 

    # --------------------------------------------------
    # 7) Symbolic regression options
    # --------------------------------------------------
    safepow(x::T, a::T) where {T<:Real} = x > zero(T) ? x^a : T(NaN)
    safesqrt(x::T) where {T<:Real} = x >= zero(T) ? sqrt(x) : T(NaN)
    safe_erfc(x::T) where {T<:Real} = begin
        if !isfinite(x)
            return T(NaN)
        end
        y = erfc(x)
        isfinite(y) ? T(y) : T(NaN)
    end
    function sigmoid_func(x::T) where {T<:Real}
        if !isfinite(x)
            return T(NaN)
        elseif x ≥ 0
            z = exp(-x)
            y = 1 / (1 + z)
        else
            z = exp(x)
            y = z / (1 + z)
        end
        return isfinite(y) ? T(y) : T(NaN)
    end
    options = Options(
        binary_operators = [+, -, *, /],
        unary_operators  = [tanh, safesqrt, sigmoid_func],

 

        complexity_of_operators = Dict(
            tanh => 1,
            safesqrt => 1,
            sigmoid_func => 2
        ),

 

        maxsize = 30,
        populations = max(Threads.nthreads(), 36),
        population_size = 160,
        batching = true,
        batch_size = 256,
        parsimony = 20f0,
    )

 

    hall_of_fame = equation_search(
        Xtrain,
        ytrain;
        niterations = 3000,
        options = options,
        parallelism = :multithreading,
        variable_names = string.(feature_cols)
    )

 

    dominating = calculate_pareto_frontier(hall_of_fame)

 

    println("\n==============================")
    println("Pareto-front equations")
    println("==============================\n")

 

    for (i, eq) in enumerate(dominating)
        println("Equation $i")
        println(eq)
        println("Tree: ", eq.tree)
        println()
    end

 

    # --------------------------------------------------
    # 8) Evaluate equations on train / val / test
    #    Parallel compute, serial collect
    # --------------------------------------------------
    neqs = length(dominating)

 

    metrics_buffer = Vector{NamedTuple}(undef, neqs)

 

    println("\n==============================")
    println("Train / Validation / Test metrics")
    println("==============================\n")

 

    Threads.@threads for i in 1:neqs
        member = dominating[i]
        tree = member.tree

 

        yhat_train = tree(Xtrain)
        yhat_val   = tree(Xval)
        yhat_test  = tree(Xtest)

 

        mtrain = finite_metrics(ytrain, yhat_train)
        mval   = finite_metrics(yval, yhat_val)
        mtest  = finite_metrics(ytest, yhat_test)

 

        metrics_buffer[i] = (
            Equation = i,
            Train_RMSE = mtrain.rmse,
            Val_RMSE = mval.rmse,
            Test_RMSE = mtest.rmse,
            Train_R2 = mtrain.r2,
            Val_R2 = mval.r2,
            Test_R2 = mtest.r2,
            NTrain = mtrain.n,
            NVal = mval.n,
            NTest = mtest.n,
            Expression = string(tree)
        )
    end

 

    results = DataFrame(metrics_buffer)

 

    for row in eachrow(results)
        println("Equation $(row.Equation)")
        println("  expression : ", row.Expression)
        println("  train RMSE : ", row.Train_RMSE)
        println("  val   RMSE : ", row.Val_RMSE)
        println("  test  RMSE : ", row.Test_RMSE)
        println("  train R²   : ", row.Train_R2)
        println("  val   R²   : ", row.Val_R2)
        println("  test  R²   : ", row.Test_R2)
        println()
    end

 

    sort!(results, :Val_RMSE)

 

    best_i = results.Equation[1]
    best_val_rmse = results.Val_RMSE[1]

 

    println("Best equation by validation RMSE = Equation $best_i")
    println("Expression: ", dominating[best_i].tree)
    println("Best validation RMSE = ", best_val_rmse)

 

    CSV.write(joinpath(outdir, "NC2_equation_ranking.csv"), results)

 

    # --------------------------------------------------
    # 9) Best equation predictions
    # --------------------------------------------------
    best_tree = dominating[best_i].tree

 

    yhat_train_best = best_tree(Xtrain)
    yhat_val_best   = best_tree(Xval)
    yhat_test_best  = best_tree(Xtest)

 

    CSV.write(
        joinpath(outdir, "NC2_train_predictions_best.csv"),
        DataFrame(Actual = ytrain, Predicted = yhat_train_best)
    )

 

    CSV.write(
        joinpath(outdir, "NC2_validation_predictions_best.csv"),
        DataFrame(Actual = yval, Predicted = yhat_val_best)
    )

 

    CSV.write(
        joinpath(outdir, "NC2_test_predictions_best.csv"),
        DataFrame(Actual = ytest, Predicted = yhat_test_best)
    )

 

    save_parity_plot(
        ytrain, yhat_train_best,
        yval, yhat_val_best,
        ytest, yhat_test_best,
        best_i,
        joinpath(outdir, "NC2_parity_best.png");
        target_name = "NC2"
    )

 

    # --------------------------------------------------
    # 10) Save parity plots and predictions for top equations
    #    Keep serial because plotting/file I/O is not reliably thread-safe
    # --------------------------------------------------
    top_k = min(5, nrow(results))

 

    for row in eachrow(results[1:top_k, :])
        eq_idx = row.Equation
        tree = dominating[eq_idx].tree

 

        yhat_train = tree(Xtrain)
        yhat_val   = tree(Xval)
        yhat_test  = tree(Xtest)

 

        CSV.write(
            joinpath(outdir, "NC2_predictions_eq$(eq_idx).csv"),
            DataFrame(
                Split = vcat(
                    fill("Train", length(ytrain)),
                    fill("Validation", length(yval)),
                    fill("Test", length(ytest))
                ),
                Actual = vcat(ytrain, yval, ytest),
                Predicted = vcat(yhat_train, yhat_val, yhat_test)
            )
        )

 

        save_parity_plot(
            ytrain, yhat_train,
            yval, yhat_val,
            ytest, yhat_test,
            eq_idx,
            joinpath(outdir, "NC2_parity_eq$(eq_idx).png");
            target_name = "N_C2"
        )
    end

 

    # --------------------------------------------------
    # 11) Residual diagnostics for best equation
    # --------------------------------------------------
    resid_test = ytest .- yhat_test_best
    resid_df = DataFrame(
        Actual = ytest,
        Predicted = yhat_test_best,
        Residual = resid_test
    )
    CSV.write(joinpath(outdir, "NC2_test_residuals_best.csv"), resid_df)

 

    test_df = df[test_idx, :]

 

    p1 = scatter(test_df.pH, resid_test, xlabel="PH", ylabel="Residual", title="Residual vs PH", markersize=3, label="")
    savefig(p1, joinpath(outdir, "Residual_vs_PH.png"))

 

    p2 = scatter(test_df.Da_CO2, resid_test, xlabel="Da_CO2", ylabel="Residual", title="Residual vs Da_CO2", markersize=3, label="")
    savefig(p2, joinpath(outdir, "Residual_vs_Da_C2.png"))

 

    p3 = scatter(test_df.kf_CO2, resid_test, xlabel="kf_CO2", ylabel="Residual", title="Residual vs kf_CO2", markersize=3, label="")
    savefig(p3, joinpath(outdir, "Residual_vs_kf_C2.png"))

 

    p4 = scatter(test_df.Keq_CO2, resid_test, xlabel="Keq_CO2", ylabel="Residual", title="Residual vs Keq_CO2", markersize=3, label="")
    savefig(p4, joinpath(outdir, "Residual_vs_Keq_C2.png"))

 

    # --------------------------------------------------
    # 12) Save equations
    # --------------------------------------------------
    open(joinpath(outdir, "NC2_symbolic_equations.txt"), "w") do io
        println(io, "Feature columns used:")
        println(io, feature_cols)
        println(io)

 

        println(io, "Best equation by validation RMSE = Equation $best_i")
        println(io, "Expression: ", dominating[best_i].tree)
        println(io)

 

        for (i, member) in enumerate(dominating)
            println(io, "Equation $i")
            println(io, member)
            println(io, "Tree: ", member.tree)
            println(io)
        end
    end

 

    # --------------------------------------------------
    # 13) Save a run summary
    # --------------------------------------------------
    best_train = finite_metrics(ytrain, yhat_train_best)
    best_val   = finite_metrics(yval, yhat_val_best)
    best_test  = finite_metrics(ytest, yhat_test_best)

 

    open(joinpath(outdir, "run_summary.txt"), "w") do io
        println(io, "Output directory: ", outdir)
        println(io, "Target: ", target)
        println(io, "Feature columns:")
        println(io, feature_cols)
        println(io)
        println(io, "Train size = ", length(ytrain))
        println(io, "Val size   = ", length(yval))
        println(io, "Test size  = ", length(ytest))
        println(io)
        println(io, "Julia threads = ", Threads.nthreads())
        println(io, "Best equation index (by validation RMSE) = ", best_i)
        println(io, "Best expression = ", dominating[best_i].tree)
        println(io)
        println(io, "Train RMSE = ", best_train.rmse)
        println(io, "Train R²   = ", best_train.r2)
        println(io, "Val RMSE   = ", best_val.rmse)
        println(io, "Val R²     = ", best_val.r2)
        println(io, "Test RMSE  = ", best_test.rmse)
        println(io, "Test R²    = ", best_test.r2)
    end

 

    println("\nSaved files in: ", outdir)
    println("  ", joinpath(outdir, "NC2_equation_ranking.csv"))
    println("  ", joinpath(outdir, "NC2_train_predictions_best.csv"))
    println("  ", joinpath(outdir, "NC2_validation_predictions_best.csv"))
    println("  ", joinpath(outdir, "NC2_test_predictions_best.csv"))
    println("  ", joinpath(outdir, "NC2_parity_best.png"))
    println("  ", joinpath(outdir, "NC2_symbolic_equations.txt"))
    println("  ", joinpath(outdir, "run_summary.txt"))
end

 

if length(ARGS) > 0
    main(ARGS[1])
else
    main("N_C2_40-02")
end

