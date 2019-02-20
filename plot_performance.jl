using CSV
using DataFrames
# using PGFPlots
using Statistics
using Plots
gr()
using StatPlots

# retrieve all the dataframes 
ids = [1, 2, 3, 4, 5, 7, 10]

single_results = [CSV.read(joinpath("single"*string(id)*"-"*string(seed), "summary.csv")) for id in ids, seed in 1:5]
multi_results = [CSV.read(joinpath("multi"*string(id)*"-"*string(seed), "summary.csv")) for id in ids, seed in 1:5]
corr_results =  [CSV.read(joinpath("corr"*string(id)*"-"*string(seed), "summary.csv")) for id in ids, seed in 1:5]

function query_metric(df::DataFrame, metric::String)
    return (mean(df[df[:variable] .== metric, :][:mean]), mean(df[df[:variable] .== metric, :][:std]))
end

function get_data(dfs::Array{DataFrame}, metric::String)
    A = zeros(size(dfs))
    for (i, df) in enumerate(dfs)
        A[i] = df[df[:variable] .== metric, :][:mean][1]
    end
    return A
end

single_success_mean = get_data(single_results, "success")
multi_success_mean = get_data(multi_results, "success")
corr_success_mean = get_data(corr_results, "success")


Plots.plot(1:7, multi_success_mean)
Plots.plot(1:7, mean(corr_success_mean, dims=2), errorbar=std(multi_success_mean, dims=2)./5)
Plots.plot(1:7, corr_success_mean)

Plots.plot(ids, mean(corr_success_mean, dims=2), label="Correction", legend=:bottomright)
Plots.plot!(ids, mean(single_success_mean, dims=2), label="Decomposition")
# Plots.plot!(ids, maximum(multi_success_mean, dims=2), label="DQN")




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
