using CSV
using DataFrames
using PGFPlots

# retrieve all the dataframes 
ids = [1, 2, 3, 4, 5, 7, 10]

single_results = [CSV.read(joinpath("single"*string(id), "summary.csv")) for id in ids]
multi_results = [CSV.read(joinpath("multi"*string(id), "summary.csv")) for id in ids]
corr_results =  [CSV.read(joinpath("corr"*string(id), "summary.csv")) for id in ids]

function query_metric(df::DataFrame, metric::String)
    return (df[df[:variable] .== metric, :][:mean][1], df[df[:variable] .== metric, :][:std][1])
end

single_success_mean, single_success_std = collect(zip([query_metric(df, "success") for df in single_results]...))
multi_success_mean, multi_success_std = collect(zip([query_metric(df, "success") for df in multi_results]...))
corr_success_mean, corr_success_std = collect(zip([query_metric(df, "success") for df in corr_results]...))

training_steps = ids .* 100_000

p = Axis([Plots.Linear(training_steps, [multi_success_mean...], legendentry="DQN"),
         Plots.Linear(training_steps, [corr_success_mean...], legendentry = "Deep Corrections")],
         xlabel="Training Budget", 
         ylabel="Percentage of successes", 
         legendPos="south west")

display(p)