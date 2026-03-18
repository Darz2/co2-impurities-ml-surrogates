using Serialization
using Plots
using Statistics

cd(@__DIR__)

# =========================
# Load saved K-fold results
# =========================
results_bubble = deserialize("outputs/kfold_results_bubble.jls")
results_dew    = deserialize("outputs/kfold_results_dew.jls")

# =========================
# Print fold summaries
# =========================
println("\n════════ Bubble CV summary ════════")
println("Validation RMSE per fold = ", results_bubble.rmse_val)
println("Validation R² per fold   = ", results_bubble.r2_val)

println("\nMean RMSE = ", mean(results_bubble.rmse_val))
println("Mean R²   = ", mean(results_bubble.r2_val))

println("\n════════ Dew CV summary ════════")
println("Validation RMSE per fold = ", results_dew.rmse_val)
println("Validation R² per fold   = ", results_dew.r2_val)

println("\nMean RMSE = ", mean(results_dew.rmse_val))
println("Mean R²   = ", mean(results_dew.r2_val))

# =========================
# Fold metric plots: Bubble
# =========================
folds = 1:length(results_bubble.rmse_val)

p1 = plot(
    folds,
    results_bubble.rmse_val,
    marker = :circle,
    lw = 2,
    xlabel = "Fold",
    ylabel = "Validation RMSE",
    title = "P_bubble: RMSE across 5 folds",
    label = "RMSE"
)

savefig(p1, "outputs/kfold_rmse_bubble.png")
println("Saved: outputs/kfold_rmse_bubble.png")

# =========================
# Fold metric plots: Bubble R²
# =========================
p2 = plot(
    folds,
    results_bubble.r2_val,
    marker = :circle,
    lw = 2,
    xlabel = "Fold",
    ylabel = "Validation R²",
    title = "P_bubble: R² across 5 folds",
    label = "R²"
)

savefig(p2, "outputs/kfold_r2_bubble.png")
println("Saved: outputs/kfold_r2_bubble.png")

# =========================
# Fold metric plots: Dew RMSE
# =========================
p3 = plot(
    folds,
    results_dew.rmse_val,
    marker = :circle,
    lw = 2,
    xlabel = "Fold",
    ylabel = "Validation RMSE",
    title = "P_dew: RMSE across 5 folds",
    label = "RMSE"
)

savefig(p3, "outputs/kfold_rmse_dew.png")
println("Saved: outputs/kfold_rmse_dew.png")

# =========================
# Fold metric plots: Dew R²
# =========================
p4 = plot(
    folds,
    results_dew.r2_val,
    marker = :circle,
    lw = 2,
    xlabel = "Fold",
    ylabel = "Validation R²",
    title = "P_dew: R² across 5 folds",
    label = "R²"
)

savefig(p4, "outputs/kfold_r2_dew.png")
println("Saved: outputs/kfold_r2_dew.png")