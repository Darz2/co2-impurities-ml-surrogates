# Usage:  julia parity_plot_PT.jl [source_id]
#
#   xvals  = gamma_cDFT  (reference / simulation)  → x-axis
#   yvals  = gamma_base  (WSD model)               → y-axis
#   colorbar = residual (yvals − xvals), diverging red→white→blue

SOURCE_ID = isempty(ARGS) ? 1 : parse(Int, ARGS[1])

using CSV, DataFrames, Plots, Statistics, JSON3
gr()   # GR backend — required for marker_z colorbar

cd(@__DIR__)
outdir = "outputs/source$(SOURCE_ID)"
mkpath(outdir)

# ── CO2 critical properties from JSON ────────────────────────
crit    = JSON3.read(read("../Pure_fluid/component_Tc.json", String))
Tc_CO2  = Float64(crit["carbon_dioxide"]["Tc"])   # [K]
Pc_CO2  = Float64(crit["carbon_dioxide"]["Pc"])   # [bar]

# ── load data ─────────────────────────────────────────────────
df      = CSV.read("../../CombinedDatasetSEC_A4.csv", DataFrame; normalizenames=true)
df      = filter(row -> row.source_id == SOURCE_ID, df)

# ---- Trial function to check -----
df[!, :Tr_CO2] = df[!, :T] ./ Tc_CO2
df[!, :Pr_CO2] = df[!, :P] ./ Pc_CO2
df[!, :Pr_over_Tr_CO2] = df[!, :Pr_CO2] ./ df[!, :Tr_CO2]

# ── compute xvals and yvals ───────────────────────────────────
df[!, :gamma_base] = df[!, :gamma_wsd_UC]   # uncorrected WSD model
df[!, :gamma_cDFT] = df[!, :gamma_wsd_UC] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]
# trial correction coefficients
a = 0.0
b = 0.0

df[!, :eps_trial] = a .+ b .* df[!, :Pr_over_Tr_CO2]
df[!, :gamma_trial] = df[!, :gamma_base] .* (1 .+ df[!, :eps_trial])

xvals = df[!, :gamma_cDFT]          # reference / simulation
yvals = df[!, :gamma_trial]          # WSD model

# yvals = df[!, :gamma_base]          # WSD model

resid = yvals .- xvals              # residual → drives colorbar
clim  = maximum(abs, resid)         # symmetric limits
P     = df[!, :P]                   # pressure  [bar]
T     = df[!, :T]                   # temperature [K]

# ── metrics ───────────────────────────────────────────────────
rmse_val = sqrt(mean(resid .^ 2))
r2_val   = 1 - sum(resid .^ 2) / sum((xvals .- mean(xvals)).^2)
println("RMSE = $(round(rmse_val, sigdigits=4))  |  R² = $(round(r2_val, sigdigits=4))")

lo = min(minimum(xvals), minimum(yvals))
hi = max(maximum(xvals), maximum(yvals))
metrics_txt = "RMSE=$(round(rmse_val, sigdigits=2))  R²=$(round(r2_val, sigdigits=3))"

# ── helper ────────────────────────────────────────────────────
function make_parity(color_vals, cb_title, plot_title, savepath)
    p = scatter(xvals, yvals;
        marker_z          = color_vals,
        color             = cgrad(:RdBu, rev=true),
        clims             = (minimum(abs, color_vals), maximum(abs, color_vals)),
        colorbar          = true,
        colorbar_title    = cb_title,
        markersize        = 5,
        markerstrokewidth = 0.3,
        alpha             = 0.75,
        xlabel            = "γ_cDFT  (xvals)",
        ylabel            = "γ_base  (yvals)",
        title             = plot_title,
        legend            = false,
        size              = (820, 620),
        right_margin      = 15Plots.mm,
        bottom_margin     = 5Plots.mm,
    )
    plot!(p, [lo, hi], [lo, hi]; lw=2, color=:black, linestyle=:dash)
    annotate!(p, lo + 0.55*(hi-lo), hi - 0.05*(hi-lo), text(metrics_txt, 9, :left))
    display(p)
    savefig(p, savepath)
    println("Saved → $savepath")
end

# ── Plot 1: coloured by Pressure ──────────────────────────────
make_parity(P, "P [bar]",
    "Parity Plot — coloured by P  (source_id=$(SOURCE_ID))",
    "$outdir/parity_plot_colored_P.png")

# ── Plot 2: coloured by Temperature ───────────────────────────
make_parity(T, "T [K]",
    "Parity Plot — coloured by T  (source_id=$(SOURCE_ID))",
    "$outdir/parity_plot_colored_T.png")
    display(T)
