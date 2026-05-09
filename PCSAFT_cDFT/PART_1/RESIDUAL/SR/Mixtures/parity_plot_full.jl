# Usage:  julia parity_plot_full.jl
#
#   xvals  = gamma_cDFT  (reference / simulation)  → x-axis
#   yvals  = gamma_trial (WSD model + trial correction) → y-axis
#   colorbar coloured by P [bar] and T [K]
#   Uses the FULL dataset (all source_ids)

using CSV, DataFrames, Plots, Statistics, JSON3
gr()   # GR backend — required for marker_z colorbar

cd(@__DIR__)
outdir = "outputs/full"
mkpath(outdir)

# ── CO2 critical properties from JSON ────────────────────────
crit   = JSON3.read(read("../Pure_fluid/component_Tc.json", String))
Tc_CO2 = Float64(crit["carbon_dioxide"]["Tc"])   # [K]
Pc_CO2 = Float64(crit["carbon_dioxide"]["Pc"])   # [bar]

# ── load full dataset (all source_ids) ───────────────────────
df = CSV.read("../../CombinedDatasetSEC_A4.csv", DataFrame; normalizenames=true)
println("Total rows: ", nrow(df))

# ── compute gamma_base and gamma_cDFT ────────────────────────
df[!, :gamma_base] = df[!, :gamma_wsd_UC]   # uncorrected WSD model
df[!, :gamma_cDFT] = df[!, :gamma_wsd_UC] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]

# ---- Trial function to check -----
df[!, :Tr_CO2]         = df[!, :T] ./ Tc_CO2
df[!, :Pr_CO2]         = df[!, :P] ./ Pc_CO2
df[!, :Pr_over_Tr_CO2] = df[!, :Pr_CO2] ./ df[!, :Tr_CO2]

# trial correction coefficients
a = 0.0
b = -0.35

df[!, :eps_trial]   = a .+ b .* df[!, :Pr_over_Tr_CO2]
df[!, :gamma_trial] = df[!, :gamma_base] .* (1 .+ df[!, :eps_trial])

xvals_raw = df[!, :gamma_cDFT]
yvals_raw = df[!, :gamma_trial]
# yvals_raw = df[!, :gamma_base]
P_raw     = df[!, :P]
T_raw     = df[!, :T]

# drop rows with NaN/Inf so metrics and plot are clean
mask  = coalesce.(isfinite.(xvals_raw) .& isfinite.(yvals_raw), false)
xvals = xvals_raw[mask]
yvals = yvals_raw[mask]
P     = P_raw[mask]
T     = T_raw[mask]
resid = yvals .- xvals

println("Rows after dropping non-finite: $(sum(mask)) / $(length(mask))")

# ── metrics ───────────────────────────────────────────────────
rmse_val = sqrt(mean(resid .^ 2))
r2_val   = 1 - sum(resid .^ 2) / sum((xvals .- mean(xvals)).^2)
println("RMSE = $(round(rmse_val, sigdigits=4))  |  R² = $(round(r2_val, sigdigits=4))")

lo = min(minimum(xvals), minimum(yvals))
hi = max(maximum(xvals), maximum(yvals))
metrics_txt = "RMSE=$(round(rmse_val, sigdigits=2))  R²=$(round(r2_val, sigdigits=3))  N=$(sum(mask))"

# ── helper ────────────────────────────────────────────────────
function make_parity(color_vals, cb_title, plot_title, savepath)
    p = scatter(xvals, yvals;
        marker_z          = color_vals,
        color             = cgrad(:RdBu, rev=true),
        clims             = (minimum(abs, color_vals), maximum(abs, color_vals)),
        colorbar          = true,
        colorbar_title    = cb_title,
        markersize        = 4,
        markerstrokewidth = 0.2,
        alpha             = 0.6,
        xlabel            = "γ_cDFT  (xvals)",
        ylabel            = "γ_trial  (yvals)",
        title             = plot_title,
        legend            = false,
        size              = (820, 620),
        right_margin      = 15Plots.mm,
        bottom_margin     = 5Plots.mm,
    )
    plot!(p, [lo, hi], [lo, hi]; lw=2, color=:black, linestyle=:dash)
    annotate!(p, lo + 0.02*(hi-lo), lo + 0.06*(hi-lo), text(metrics_txt, 9, :left))
    display(p)
    savefig(p, savepath)
    println("Saved → $savepath")
end

# ── Plot 1: coloured by Pressure ──────────────────────────────
make_parity(P, "P [bar]",
    "Parity Plot — full dataset — coloured by P",
    "$outdir/parity_plot_colored_P.png")

# ── Plot 2: coloured by Temperature ───────────────────────────
make_parity(T, "T [K]",
    "Parity Plot — full dataset — coloured by T",
    "$outdir/parity_plot_colored_T.png")
