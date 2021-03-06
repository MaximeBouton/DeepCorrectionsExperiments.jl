using CSV
using DataFrames
using Statistics
using Plots
pgfplots()
using StatPlots

# retrieve all the dataframes 
ids = [1, 2, 3, 4, 5, 7, 10]

training_budget = [100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 550_000]

single_results = [CSV.read(joinpath("single"*string(id), "summary.csv")) for id in ids]
multi_results = [CSV.read(joinpath("multi"*string(id), "summary.csv")) for id in [10,15,20,25,30,40,55]]
corr_results = [CSV.read(joinpath("corr"*string(id), "summary.csv")) for id in ids]
vdn_results = [CSV.read(joinpath("vdn"*string(id), "summary.csv")) for id in [10,15,20,25,30,40,55]]

function query_metric(df::DataFrame, metric::String, op)
    return (op(df[df[:variable] .== metric, :][:mean]), op(df[df[:variable] .== metric, :][:std]))
end

function get_data(dfs::Array{DataFrame}, metric::String, field::Symbol)
    A = Vector{Vector{Float64}}(undef, length(dfs))
    for (i, df) in enumerate(dfs)
        A[i] = df[df[:variable] .== metric, :][field]
    end
    return A
end

metric = "success"

function compare_plot(single_results::Array{DataFrame}, 
                      multi_results::Array{DataFrame}, 
                      corr_results::Union{Nothing, Array{DataFrame}},
                      vdn_results::Array{DataFrame},
                      metric::String, op)
    single_mean = get_data(single_results, metric, :mean)
    single_std = get_data(single_results, metric, :std)
    multi_mean = get_data(multi_results, metric, :mean)
    multi_std = get_data(multi_results, metric, :std)
    corr_success_mean = get_data(corr_results, metric, :mean)
    corr_std = get_data(corr_results, metric, :std)
    vdn_mean =  get_data(vdn_results, metric, :mean)  
    vdn_std = get_data(vdn_results, metric, :std)
    p = Plots.plot(training_budget, op.(single_mean), 
                   label="Decomposition", 
                   xlabel="Training budget (number of interactions)", 
                   ylabel="Success rate",
                   color = :black,
                   legend=:bottomright,
                   legendfontsize=16,
                   titlefontsize=16,
                   tickfontsize=16,
                    markershape = :utriangle,
                    markersize = 5,
                    markercolor = :black,
                    markerstrokecolor = :black,
                   linewidth=2,
                   errorbar=op.(single_std)/length(single_std))
    Plots.plot!(p, training_budget, op.(corr_success_mean), 
                   label="Correction", 
                   color=:red,
                   markershape = :square,
                    markersize = 5,
                    markercolor = :red,
                    markerstrokecolor = :red,
                    linewidth=2,
                    errorbar=op.(corr_std)/length(corr_std)),
    Plots.plot!(p, training_budget, op.(multi_mean), 
                   label="DQN",
                   color=:blue,
                   markershape = :circle,
                    markersize = 5,
                    markercolor = :blue,
                    markerstrokecolor = :blue,
                    linewidth=2,
                    errorbar=op.(multi_std)/length(multi_std))
    Plots.plot!(p, training_budget, op.(vdn_mean), 
                   label="VDN",
                   color=:orange,
                   markershape = :diamond,
                    markersize = 5,
                    markercolor = :orange,
                    markerstrokecolor = :orange,
                    linewidth=2,
                    errorbar=op.(vdn_std)/length(vdn_std))
    return p
end

p = compare_plot(single_results, multi_results, corr_results, vdn_results, "success", x->100*mean(x))

#=
single_success_mean, single_success_std = collect(zip([query_metric(df, "success") for df in single_results]...))



multi_success_mean, multi_success_std = collect(zip([query_metric(df, "success") for df in multi_results]...))
corr_success_mean, corr_success_std = collect(zip([query_metric(df, "success") for df in corr_results]...))

training_steps = ids .* 100_000

p = Axis([Plots.Linear(training_steps, [multi_success_mean...], legendentry="DQN"),
         Plots.Linear(training_steps, [corr_success_mean...], legendentry = "Deep Corrections"),
         Plots.Linear(training_steps, [single_success_mean...], legendentry = "Decomposition")],
         xlabel="Training Budget", 
         ylabel="Percentage of successes", 
         legendPos="south west")

display(p)
=#