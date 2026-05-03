using Serialization
using SymbolicRegression
using Statistics
using Plots
gr()   # GR backend — required for marker_z colorbar

# ============================================================
# Metrics
# ============================================================

rmse(y, yhat) = sqrt(mean((y .- yhat).^2))

function r2_score(y, yhat)
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    return 1 - ss_res / ss_tot
end

function print_metrics(name, y, yhat)
    println("\n══ Metrics: $name ══")
    println("RMSE = ", rmse(y, yhat))
    println("R²   = ", r2_score(y, yhat))
end

# ============================================================
# Custom operators
# ============================================================

safepow(x::T, a::T) where {T<:Real} = x > zero(T) ? x^a : T(NaN32)
safe_div(x, y) = x / (abs(y) > 1e-12 ? y : 1e-12)
safe_sqrt(x::T) where {T<:Real} = sqrt(abs(x))
# ============================================================
# Render options
# Must match training operators
# ============================================================

function get_render_options()
    return Options(
        binary_operators        = (+, -, *, safepow),
        unary_operators         = (safe_sqrt, abs),
        complexity_of_operators = [safepow => 3, abs => 2],
        complexity_of_constants = 2,
    )
end

# ============================================================
# Hall of fame utilities
# ============================================================

load_hall_of_fame(filepath::AbstractString) = deserialize(String(filepath))

function get_equation_string(member, options)
    return string_tree(member.tree, options)
end

# ============================================================
# String utilities
# ============================================================

function round_numbers(eq_str::AbstractString; digits=5)
    s = String(eq_str)
    return replace(s, r"-?\d+\.\d+(?:[eE][+-]?\d+)?" => m -> begin
        num = parse(Float64, m)
        if abs(num) < 1e-12
            "0"
        else
            str = string(round(num; sigdigits=digits))
            if occursin(".", str) && !occursin("e", lowercase(str))
                str = rstrip(rstrip(str, '0'), '.')
            end
            (isempty(str) || str == "-") ? "0" : str
        end
    end)
end

function find_matching_paren(str::AbstractString, start_pos::Int)
    s = String(str)
    depth = 1
    i = start_pos + 1
    while i <= lastindex(s) && depth > 0
        if s[i] == '('
            depth += 1
        elseif s[i] == ')'
            depth -= 1
        end
        i += 1
    end
    return depth == 0 ? i - 1 : -1
end

function extract_function_args(str::AbstractString, func_name::AbstractString, is_binary::Bool)
    s = String(str)
    fname = String(func_name)

    match_pos = findfirst(fname * "(", s)
    match_pos === nothing && return nothing

    start = first(match_pos)
    open_paren = start + length(fname)
    close_paren = find_matching_paren(s, open_paren)
    close_paren == -1 && return nothing

    inner = s[open_paren+1:close_paren-1]

    if is_binary
        depth = 0
        comma_pos = 0
        for (i, c) in enumerate(inner)
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
            elseif c == ',' && depth == 0
                comma_pos = i
                break
            end
        end
        comma_pos == 0 && return nothing

        arg1 = strip(inner[1:comma_pos-1])
        arg2 = strip(inner[comma_pos+1:end])
        return (arg1, arg2, start, close_paren)
    else
        return (strip(inner), start, close_paren)
    end
end

function strip_outer_parens(s::AbstractString)
    out = strip(String(s))
    while startswith(out, "(") && endswith(out, ")")
        close_pos = find_matching_paren(out, firstindex(out))
        if close_pos == lastindex(out)
            out = strip(out[2:end-1])
        else
            break
        end
    end
    return out
end

function clean_equation_string(s::AbstractString)
    out = String(s)
    out = replace(out, "+ -" => "- ")
    out = replace(out, "- -" => "+ ")
    out = replace(out, r"\s+" => " ")
    return strip(out)
end

# ============================================================
# Equation rendering
# ============================================================

function string_to_math(eq_str::AbstractString; digits=5)
    math = round_numbers(eq_str; digits=digits)

    for _ in 1:200
        result = extract_function_args(math, "safepow", true)
        result === nothing && break
        arg1, arg2, start, end_pos = result
        arg1 = strip_outer_parens(arg1)
        arg2 = strip_outer_parens(arg2)
        replacement = "($arg1)^($arg2)"
        math = math[1:start-1] * replacement * math[end_pos+1:end]
    end

    math = clean_equation_string(math)
    return math
end

function substitute_features(eq_str::AbstractString; feature_map=Dict("F1" => "(1 - Tr)"))
    out = String(eq_str)
    for (k, v) in feature_map
        out = replace(out, k => v)
    end
    return out
end

function math_to_latex(s::AbstractString)
    latex = strip(String(s))

    latex = replace(latex, "T/Tc" => "\\frac{T}{T_c}")
    latex = replace(latex, "Tr"   => "T_r")
    latex = replace(latex, "F1"   => "F_1")

    for _ in 1:200
        idx = findfirst(")^(", latex)
        idx === nothing && break

        close_base = first(idx)
        open_exp   = last(idx)

        # walk backwards from ')' before ^ to find matching '('
        depth = 0; open_base = 0
        for i in close_base:-1:firstindex(latex)
            if latex[i] == ')'; depth += 1
            elseif latex[i] == '('; depth -= 1
                if depth == 0; open_base = i; break; end
            end
        end
        open_base == 0 && break

        # walk forwards from '(' after ^ to find matching ')'
        depth = 0; close_exp = 0
        for i in open_exp:lastindex(latex)
            if latex[i] == '('; depth += 1
            elseif latex[i] == ')'; depth -= 1
                if depth == 0; close_exp = i; break; end
            end
        end
        close_exp == 0 && break

        base = strip_outer_parens(strip(latex[open_base+1:close_base-1]))
        expo = strip_outer_parens(strip(latex[open_exp+1:close_exp-1]))

        repl = "\\left($base\\right)^{$expo}"
        latex = latex[1:open_base-1] * repl * latex[close_exp+1:end]
    end

    latex = replace(latex, "*" => " ")
    latex = replace(latex, "+ -" => "- ")
    latex = replace(latex, "- -" => "+ ")
    latex = replace(latex, r"\s+" => " ")

    return strip(latex)
end

function string_to_latex(eq_str::AbstractString; digits=5, feature_map=Dict("F1" => "(1 - Tr)"))
    math = string_to_math(eq_str; digits=digits)
    math = substitute_features(math; feature_map=feature_map)
    latex = math_to_latex(math)
    return latex
end

function render_hall_of_fame(filepath::AbstractString; n_best=5, digits=5, feature_map=Dict("F1" => "(1 - Tr)"))
    hof = load_hall_of_fame(filepath)
    options = get_render_options()

    dominating = calculate_pareto_frontier(hof)
    n_display = min(n_best, length(dominating))

    println("\nTop $n_display equations from Pareto frontier:")
    println("="^80)

    for i in 1:n_display
        member = dominating[i]
        eq_str = get_equation_string(member, options)
        complexity = compute_complexity(member, options)

        println("\n[$i] Complexity: $complexity, Loss: $(round(member.loss, sigdigits=6))")
        println("-"^60)
        println(string_to_latex(eq_str; digits=digits, feature_map=feature_map))
    end

    println("="^80)
end

function export_hall_of_fame_tex(
    filepath::AbstractString,
    output_tex::AbstractString;
    n_best=10,
    digits=5,
    feature_map=Dict("F1" => "(1 - Tr)"),
    title="Symbolic Regression Equations"
)
    hof = load_hall_of_fame(filepath)
    options = get_render_options()
    dominating = calculate_pareto_frontier(hof)
    n_display = min(n_best, length(dominating))

    open(String(output_tex), "w") do io
        println(io, raw"\documentclass[11pt]{article}")
        println(io, raw"\usepackage{amsmath}")
        println(io, raw"\usepackage[a4paper,margin=1in]{geometry}")
        println(io, raw"\begin{document}")
        println(io, "\\section*{$title}")
        println(io, raw"\[ T_r = \frac{T}{T_c}, \qquad F_1 = 1 - T_r \]")
        println(io, "")

        for i in 1:n_display
            member = dominating[i]
            eq_str = get_equation_string(member, options)
            complexity = compute_complexity(member, options)
            loss = round(member.loss, sigdigits=6)
            eq_latex = string_to_latex(eq_str; digits=digits, feature_map=feature_map)

            println(io, "\\subsection*{Equation $i}")
            println(io, "Complexity: $complexity\\\\")
            println(io, "Loss: $loss")
            println(io, raw"\[")
            println(io, "\\sigma = $eq_latex")
            println(io, raw"\]")
            println(io, "")
        end

        println(io, raw"\end{document}")
    end
end

# ============================================================
# Validation-based equation selection
# ============================================================

function select_best_equation(dominating, X_val, y_val, options)
    best_idx = 0
    best_rmse_val = Inf

    println("\n══ Pareto frontier equations ══")

    for (i, member) in enumerate(dominating)
        eq_str = string_tree(member.tree, options)
        complexity = compute_complexity(member, options)

        yhat_val_tmp, ok_val_tmp = eval_tree_array(member.tree, X_val', options)

        if ok_val_tmp
            this_rmse_val = rmse(y_val, yhat_val_tmp)
            println("[$i] Complexity = $complexity, Loss = $(member.loss), Val RMSE = $this_rmse_val")
            println("    ", eq_str)

            if this_rmse_val < best_rmse_val
                best_rmse_val = this_rmse_val
                best_idx = i
            end
        else
            println("[$i] Complexity = $complexity, Loss = $(member.loss), Val RMSE = failed")
            println("    ", eq_str)
        end
    end

    if best_idx == 0
        error("No valid equation found on Pareto frontier.")
    end

    return best_idx, best_rmse_val
end

# ============================================================
# Parity plot
# ============================================================

function parity_plot(
    y_train, yhat_train,
    y_val,   yhat_val,
    y_test,  yhat_test;
    title    = "Parity Plot",
    xlabel   = "Actual",
    ylabel   = "Predicted",
    eq_idx   = nothing,
    savepath = nothing,
)
    rmse_train = rmse(y_train, yhat_train)
    rmse_val   = rmse(y_val,   yhat_val)
    rmse_test  = rmse(y_test,  yhat_test)
    r2_train   = r2_score(y_train, yhat_train)
    r2_val     = r2_score(y_val,   yhat_val)
    r2_test    = r2_score(y_test,  yhat_test)

    all_actual    = vcat(y_train,    y_val,    y_test)
    all_predicted = vcat(yhat_train, yhat_val, yhat_test)
    lo = min(minimum(all_actual), minimum(all_predicted))
    hi = max(maximum(all_actual), maximum(all_predicted))

    # residuals for diverging colorbar (red = over-predicted, blue = under-predicted)
    res_train = yhat_train .- y_train
    res_val   = yhat_val   .- y_val
    res_test  = yhat_test  .- y_test
    clim = maximum(abs, vcat(res_train, res_val, res_test))

    p = scatter(y_train, yhat_train;
                marker_z=res_train, color=cgrad(:RdBu, rev=true),
                clims=(-clim, clim), colorbar=true,
                colorbar_title="Residual (Predicted − Actual)",
                label="Train", markersize=4, markerstrokewidth=0.3,
                xlabel=xlabel, ylabel=ylabel, title=title, legend=:topleft,
                size=(820, 620), right_margin=15Plots.mm, bottom_margin=5Plots.mm)
    scatter!(p, y_val,  yhat_val;
                marker_z=res_val,   color=cgrad(:RdBu, rev=true),
                clims=(-clim, clim), colorbar=true, label="Validation",
                markersize=4, markerstrokewidth=0.3, markershape=:diamond)
    scatter!(p, y_test, yhat_test;
                marker_z=res_test,  color=cgrad(:RdBu, rev=true),
                clims=(-clim, clim), colorbar=true, label="Test",
                markersize=4, markerstrokewidth=0.3, markershape=:utriangle)
    plot!(p, [lo, hi], [lo, hi]; label="y = x", lw=2, color=:black, linestyle=:dash)

    ax = lo + 0.52*(hi - lo)
    rows = String[]
    eq_idx !== nothing && push!(rows, "Eq. index = $eq_idx")
    push!(rows, "Train: RMSE=$(round(rmse_train, sigdigits=2))  R²=$(round(r2_train, sigdigits=3))")
    push!(rows, "Val:   RMSE=$(round(rmse_val,   sigdigits=2))  R²=$(round(r2_val,   sigdigits=3))")
    push!(rows, "Test:  RMSE=$(round(rmse_test,  sigdigits=2))  R²=$(round(r2_test,  sigdigits=3))")
    for (k, row) in enumerate(rows)
        annotate!(p, ax, hi - (0.68 + (k-1)*0.06)*(hi - lo), text(row, 9, :left))
    end

    if savepath !== nothing
        savefig(p, savepath)
        println("Saved parity plot: $savepath")
    end

    return p
end