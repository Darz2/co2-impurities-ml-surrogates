using CSV
using DataFrames
using Plots
plotlyjs()

cd(@__DIR__)

# ── Sources to plot (up to 20) ───────────────────────────────────────────────
SOURCES = [1]          # change range or list, e.g. [1,3,5,7] for specific IDs

# 20 visually distinct colours (ColorBrewer + custom)
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#018ce1",
]

# ── Load full dataset once ───────────────────────────────────────────────────
df_all = CSV.read("1691761_CombinedDataset_A3.csv", DataFrame; normalizenames=true)

mkpath("outputs/multi_source")

# ── Build plot — first scatter initialises the 3-D canvas ────────────────────
p = nothing

for (i, sid) in enumerate(SOURCES)
    df = filter(row -> row.source_id == sid, df_all)
    isempty(df) && (println("source_id $sid — no rows, skipping"); continue)

    # Compute eps_base
    r1                 = (df[!, :rhoL_carbon_dioxide] .- df[!, :rhoV_carbon_dioxide]) ./
                         (df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide])
    df[!, :gamma_base] = df[!, :gamma0_carbon_dioxide] .* r1 .^ 2
    df[!, :gamma_cDFT] = df[!, :gamma_wsd] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]
    df[!, :eps_base]   = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

    keep = isfinite.(df[!, :T]) .& isfinite.(df[!, :P]) .& isfinite.(df[!, :eps_base])
    df   = df[keep, :]
    println("source_id $sid — $(nrow(df)) rows")

    Tr  = df[!, :T] ./ 309.147
    Pr  = df[!, :P] ./ 82.379
    eps = df[!, :eps_base]
    eps_scaled = eps .* (1 .+ (-0.35) .* Pr ./ Tr)

    col = PALETTE[mod1(i, length(PALETTE))]

    common = (;
        markerstrokecolor = :black,
        markerstrokewidth = 0.75,
        markersize        = 2,
        colorbar          = false,
        xlabel            = "T / Tc",
        ylabel            = "P / Pc",
        zlabel            = "ε",
        title             = "Residual ε — Sources $(first(SOURCES)) – $(last(SOURCES))",
        legend            = true,
        grid              = true,
        gridalpha         = 0.5,
        camera            = (30, 20),
        size              = (1000, 750),
    )

    if p === nothing
        p = scatter(Tr, Pr, eps;
            label       = "src $sid  raw",
            color       = col,
            markercolor = col,
            common...,
        )
    else
        scatter!(p, Tr, Pr, eps;
            label       = "src $sid  raw",
            color       = col,
            markercolor = col,
            common...,
        )
    end

    scatter!(p, Tr, Pr, eps_scaled;
        label             = "src $sid  scaled",
        color             = "#2ca02c",
        markercolor       = "#2ca02c",
        markerstrokecolor = :black,
        markerstrokewidth = 0.75,
        markersize        = 2,
        colorbar          = false,
    )
end

savefig(p, "outputs/multi_source/residual_3d_multi.html")
savefig(p, "outputs/multi_source/residual_3d_multi.png")
println("Saved → outputs/multi_source/residual_3d_multi.html")
println("Saved → outputs/multi_source/residual_3d_multi.png")

display(p)
