include("common.jl")
#include("paper_plots.jl")
using DataFrames
using GLM
using Printf
using Random
using SparseArrays
using Statistics
using Pkg
using Plots.PlotMeasures
using StatsBase
using Distributed
using ScHoLP
using MLDataUtils

#@load DecisionTreeClassifier

#Pkg.add("Plots")
#Pkg.add("ColorSchemes")
#using Plots
using Plots
using ColorSchemes
using CSV
using PyPlot
using PyCall
using FileIO
using JLD2
using MLJ

#@pyimport matplotlib.patches as patch
patch = pyimport("matplotlib.patches")

using ScikitLearn
@sk_import linear_model: LogisticRegression
#using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, make_pipeline
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.GridSearch: GridSearchCV

#@sk_import tree: DecisionTreeClassifier
#@sk_import model_selection: train_test_split
#Pkg.add("DataStructures")

#import ScikitLearn: fit!
#import MLDataUtils: predict
#using ScikitLearn: fit!, predict

import DataStructures
using DataStructures: counter
using Combinatorics

using ScikitLearn
@sk_import linear_model: LogisticRegression

#global ego_types = ["default", "contracted", "star", "expanded"]
global ego_types = ["default", "contracted", "star"]

# Construct HONData for a given ego
function egonet_dataset(dataset::HONData, ego::Int64, B::SpIntMat)
    in_egonet = zeros(Bool, size(B, 1))
    in_egonet[ego] = true
    in_egonet[findnz(B[:, ego])[1]] .= true

    node_map = Dict{Int64, Int64}()
    function get_key(x::Int64)
        if haskey(node_map, x); return node_map[x]; end
        n = length(node_map) + 1
        node_map[x] = n
        return n
    end
    ego_key = get_key(ego)
    
    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]
    curr_ind = 1
    for (nvert, time) in zip(dataset.nverts, dataset.times)
        end_ind = curr_ind + nvert - 1
        simplex = dataset.simplices[curr_ind:end_ind]
        curr_ind += nvert
        simplex_in_egonet = [v for v in simplex if in_egonet[v]]
        if length(simplex_in_egonet) > 0
            mapped_simplex = [get_key(v) for v in simplex_in_egonet]
            append!(new_simplices, mapped_simplex)
            push!(new_nverts, length(mapped_simplex))
            push!(new_times, time)
        end
    end

    return HONData(new_simplices, new_nverts, new_times, "egonet")
end

# Construct SortedDict (time->simplices) for a given ego, and return degree and length of ego
function get_egonet_data_dict(dataset::HONData, ego::Int64, B::SpIntMat)
    in_egonet = zeros(Bool, size(B, 1))
    in_egonet[ego] = true
    in_egonet[findnz(B[:, ego])[1]] .= true

    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]
    curr_ind = 1
    nodes = []
    start = false
    cont_len = 0
    def_len = 0
    exp_len = 0
    star_len = 0
    degree = 0
    broke = false
    global_min = 0

    for (nvert, time) in zip(dataset.nverts, dataset.times)
        end_ind = curr_ind + nvert - 1
        simplex = dataset.simplices[curr_ind:end_ind]
        simplex_in_egonet = []
        expanded_simplex_in_egonet = []
        curr_ind += nvert

        simplex_in_egonet = [v for v in simplex if in_egonet[v]]

        if simplex == simplex_in_egonet && length(simplex_in_egonet) > 1
            def_len += 1
        end

        # Check Expanded
        for v in simplex
            if in_egonet[v]
                expanded_simplex_in_egonet = simplex
                break
            end
        end

        if length(expanded_simplex_in_egonet) > 1
            exp_len += 1
        end
        
        if length(simplex_in_egonet) > 1
            # Check Star
            if ego in simplex_in_egonet
                star_len += 1
            end

            cont_len += 1
        end
    end
    degree = star_len
    return degree, def_len, cont_len, star_len, exp_len, global_min
end

function get_egonet_data_dict_two(dataset::HONData, ego::Int64, B::SpIntMat, ego_type::String)
    sorted_data = DataStructures.SortedDict()

    in_egonet = zeros(Bool, size(B, 1))
    in_egonet[ego] = true
    in_egonet[findnz(B[:, ego])[1]] .= true

    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]
    curr_ind = 1
    nodes = []
    start = false

    for (nvert, time) in zip(dataset.nverts, dataset.times)
        end_ind = curr_ind + nvert - 1
        simplex = dataset.simplices[curr_ind:end_ind]
        simplex_in_egonet = []
        curr_ind += nvert
        #base_data[time] = []        

        simplex_in_egonet = [v for v in simplex if in_egonet[v]]

        #should only get called on first iteration of outer for loop?
        if !(haskey(sorted_data, time))
            sorted_data[time] = []
        end

        if (ego_type == "default")
            broke = false
            if simplex != simplex_in_egonet
                broke = true
            end
            if (broke == true)
                continue
            end
        end

        if (ego_type == "expanded")
            for v in simplex
                if in_egonet[v]
                    simplex_in_egonet = simplex
                    break
                end
            end
        end

        if length(simplex_in_egonet) > 1 # changed from 0
            if (ego_type == "star")
                if !(ego in simplex_in_egonet)
                    continue
                end
            end

            append!(new_simplices, simplex_in_egonet)
            push!(new_nverts, length(simplex_in_egonet))
            push!(new_times, time)
            Base.GC.enable(false)
            #sorted_data[time] = simplex_in_egonet
            push!(sorted_data[time], simplex_in_egonet)
            Base.GC.enable(true)
        end
    end
    global_min = minimum(collect(keys(sorted_data)))
    return sorted_data, global_min
end

# Construct SortedDict (time->simplices) for a given ego, and return degree and length of ego
function get_egonet_data_dict_with_data(dataset::HONData, ego::Int64, B::SpIntMat)
    in_egonet = zeros(Bool, size(B, 1))
    in_egonet[ego] = true
    in_egonet[findnz(B[:, ego])[1]] .= true

    contracted_sorted_data, def_sorted_data, star_sorted_data = 
        [DataStructures.SortedDict() for _ = 1:3]

    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]
    curr_ind = 1
    nodes = []
    start = false
    cont_len = 0
    def_len = 0
    star_len = 0
    degree = 0
    broke = false
    global_min = 0

    for (nvert, time) in zip(dataset.nverts, dataset.times)
        end_ind = curr_ind + nvert - 1
        simplex = dataset.simplices[curr_ind:end_ind]
        simplex_in_egonet = []
        curr_ind += nvert

        simplex_in_egonet = [v for v in simplex if in_egonet[v]]

        if simplex == simplex_in_egonet && length(simplex_in_egonet) > 1
            def_len += 1
            try
                push!(def_sorted_data[time], simplex_in_egonet)
            catch e
                def_sorted_data[time] = [simplex_in_egonet]
            end
        end
        
        if length(simplex_in_egonet) > 1
            # Check Star
            if ego in simplex_in_egonet
                star_len += 1
                try
                    push!(star_sorted_data[time], simplex_in_egonet)
                catch e
                    star_sorted_data[time] = [simplex_in_egonet]
                end
            end

            cont_len += 1
            try
                push!(contracted_sorted_data[time], simplex_in_egonet)
            catch e
                contracted_sorted_data[time] = [simplex_in_egonet]
            end
        end
    end
    degree = star_len
    return def_sorted_data, contracted_sorted_data, star_sorted_data, degree, def_len, cont_len, star_len
end

function get_len_degree(dataset::HONData, ego::Int64, B::SpIntMat)
    in_egonet = zeros(Bool, size(B, 1))
    in_egonet[ego] = true
    in_egonet[findnz(B[:, ego])[1]] .= true

    contracted_sorted_data, def_sorted_data, star_sorted_data, expanded_sorted_data = 
        [DataStructures.SortedDict() for _ = 1:4]

    node_map = Dict{Int64, Int64}()
    function get_key(x::Int64)
        if haskey(node_map, x); return node_map[x]; end
        n = length(node_map) + 1
        node_map[x] = n
        return n
    end
    ego_key = get_key(ego)
    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]
    curr_ind = 1
    nodes = []
    start = false
    len = 0
    degree = 0
    broke = false

    for (nvert, time) in zip(dataset.nverts, dataset.times)
        end_ind = curr_ind + nvert - 1
        simplex = dataset.simplices[curr_ind:end_ind]
        simplex_in_egonet = []
        curr_ind += nvert

        simplex_in_egonet = [v for v in simplex if in_egonet[v]]

        if (ego in simplex)
            degree+=1
        end

        for simplex in simplex_in_egonet
            len+=length(simplex)
        end
    end
    return len, degree
end

function get_eligible_egos(dataset::HONData)
    # read data
    A1, At1, B1 = basic_matrices(dataset.simplices, dataset.nverts)

    # Get eligible egos
    n = size(B1, 1)
    tri_order = proj_graph_degree_order(B1)
    in_tri = zeros(Int64, n, Threads.nthreads())
    Threads.@threads for i = 1:n
        for (j, k) in neighbor_pairs(B1, tri_order, i)
            if B1[j, k] > 0
                tid = Threads.threadid()
                in_tri[[i, j, k], tid] .= 1
            end
        end
    end
    eligible_egos = findall(vec(sum(in_tri, dims=2)) .> 0)
    num_eligible = length(eligible_egos)
    println("$num_eligible eligible egos")

    return eligible_egos, B1
end

function get_sampled_egos(dataset::HONData, num_egos::Int64)
    
    eligible_egos, B1 = get_eligible_egos(dataset)

    # Sample from eligible egos
    sampled_egos =
        eligible_egos[StatsBase.sample(1:length(eligible_egos),
                                       num_egos, replace=false)]

    return sampled_egos, B1, eligible_egos
end

function load_egos_percentile(dataset::String, num_egos_limit::Int64, def_length_bounds::Array{Int64, 1}, cont_length_bounds::Array{Int64, 1},
    star_length_bounds::Array{Int64, 1}, exp_length_bounds::Array{Int64, 1},
    degree_bounds::Array{Int64, 1}, all::Bool = false)

    def_egos_to_return = []
    cont_egos_to_return = []
    star_egos_to_return = []
    exp_egos_to_return = []
    
    global_sorted_dict = load_sorted_dicts(dataset, false)

    if all == true
        return collect(keys(global_sorted_dict))
    end
    degrees_list = [info[1] for (ego, info) in global_sorted_dict]
    
    low_percentile = percentile(degrees_list, degree_bounds[1])
    high_percentile = percentile(degrees_list, degree_bounds[2])

    for (ego, info) in global_sorted_dict
        if low_percentile <= info[1] <= high_percentile
            if def_length_bounds[1] <= info[2] <= def_length_bounds[2]
                push!(def_egos_to_return, ego)
            end
            if cont_length_bounds[1] <= info[3] <= cont_length_bounds[2]
                push!(cont_egos_to_return, ego)
            end
            if star_length_bounds[1] <= info[4] <= star_length_bounds[2]
                push!(star_egos_to_return, ego)
            end
            if exp_length_bounds[1] <= info[5] <= exp_length_bounds[2]
                push!(exp_egos_to_return, ego)
            end
        end
    end

    #println(def_egos_to_return)
    #println(cont_egos_to_return)
    #println(star_egos_to_return)
    #println(exp_egos_to_return)

    return def_egos_to_return, cont_egos_to_return, star_egos_to_return, exp_egos_to_return
end

function load_egos(dataset::String, num_egos_limit::Int64, def_length_bounds::Array{Int64, 1}, cont_length_bounds::Array{Int64, 1},
    star_length_bounds::Array{Int64, 1}, #=exp_length_bounds::Array{Int64, 1},=#
    degree_bounds::Array{Int64, 1}, all::Bool = false)

    def_egos_to_return = []
    cont_egos_to_return = []
    star_egos_to_return = []
    exp_egos_to_return = []
    
    def_count = 0
    cont_count = 0
    star_count = 0
    exp_count = 0

    global_sorted_dict = load_sorted_dicts(dataset, false)

    if all == true
        return collect(keys(global_sorted_dict))
    end

    for (ego, info) in global_sorted_dict
        if degree_bounds[1] <= info[1] <= degree_bounds[2]
            if def_length_bounds[1] <= info[2] <= def_length_bounds[2]
                if def_count < num_egos_limit
                    push!(def_egos_to_return, ego)
                    def_count += 1
                end
            end
            if cont_length_bounds[1] <= info[3] <= cont_length_bounds[2]
                if cont_count < num_egos_limit
                    push!(cont_egos_to_return, ego)
                    cont_count += 1
                end
            end
            if star_length_bounds[1] <= info[4] <= star_length_bounds[2]
                if star_count < num_egos_limit
                    push!(star_egos_to_return, ego)
                    star_count += 1
                end
            end
            #=
            if exp_length_bounds[1] <= info[5] <= exp_length_bounds[2]
                if exp_count < num_egos_limit
                    push!(exp_egos_to_return, ego)
                    exp_count += 1
                end
            end
            =#
        end
    end

    #println(def_egos_to_return)
    #println(cont_egos_to_return)
    #println(star_egos_to_return)
    #println(exp_egos_to_return)

    return def_egos_to_return, cont_egos_to_return, star_egos_to_return#, exp_egos_to_return
end

function plot_alternetworks_simplex_size(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    X = []
    Y = []

    for len in [5, 10, 15, 20, 25, 30]
        X_len = 1:len
        Y_len = []
        print(stdout, "Length: $len \r")
        flush(stdout)
        # NEED Y RANDOM AND A COUNT TO SEE IF USER ARRIVED IN USER SIMPLEX
        alter_dict = DataStructures.SortedDict()

        for (ego, info) in global_sorted_dict
            egonet = info[3]
            alters = get_alters(egonet, ego)

            for alter in alters
                alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego) # get star alternetwork
                
                if length_alternetwork == len
                    for l in X_len
                        try
                            push!(alter_dict[l], length(get_alternetwork_simplex(alternetwork, l)))
                        catch e
                            alter_dict[l] = [length(get_alternetwork_simplex(alternetwork, l))]
                        end
                    end
                end
            end
        end

        for (k, v) in alter_dict
            #push!(X_len, k)
            push!(Y_len, mean(v))
        end

        push!(X, X_len)
        push!(Y, Y_len)
    end

    Plots.plot(X, Y, xlabel = "ordinal time", ylabel = "average incoming simplex size", xlim = (1, 30), label = ["Alternet Length 5" "Alternet Length 10" "Alternet Length 15" "Alternet Length 20" "Alternet Length 25" "Alternet Length 30"], left_margin = 10mm, bottom_margin = 10mm, legend =:topright, title = "Average incoming simplex size - Star", linewidth=0.5)
    Plots.savefig("$(dataset)_star_alternetwork_simplex_size.pdf")
end

function plot_alternetworks_novelty(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    X = []
    Y = []

    for len in [5, 10, 15, 20, 25, 30]
        print(stdout, "Length: $len \r")
        flush(stdout)
        # NEED Y RANDOM AND A COUNT TO SEE IF USER ARRIVED IN USER SIMPLEX
        alter_dict = DataStructures.SortedDict()
        X_len = []
        Y_len = []
        for (ego, info) in global_sorted_dict
            egonet = info[3]
            alters = get_alters(egonet, ego)

            for alter in alters
                alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego) # get star alternetwork
                
                if length_alternetwork == len
                    alter_dict_len = get_alternetwork_novelty(alternetwork)

                    for (k, v) in alter_dict_len
                        try
                            push!(alter_dict[k], v)
                        catch e
                            alter_dict[k] = [v]
                        end
                    end
                end
            end
        end
        
        for (k, v) in alter_dict
            push!(X_len, k)
            push!(Y_len, mean(v))
        end

        push!(X, X_len)
        push!(Y, Y_len)
    end

    

    Plots.plot(X, Y, xlabel = "ordinal time", ylabel = "average novelty", xlim = (1, 30), label = ["Alternet Length 5" "Alternet Length 10" "Alternet Length 15" "Alternet Length 20" "Alternet Length 25" "Alternet Length 30"], left_margin = 10mm, bottom_margin = 10mm, legend =:topright, title = "Average novelty - Star", linewidth=0.5)
    Plots.savefig("$(dataset)_star_alternetwork_novelty.pdf")
end

function plot_alternetworks_n(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    X = []
    Y = []

    for len in [5, 10, 15, 20, 25, 30]
        print(stdout, "Length: $len \r")
        flush(stdout)
        # NEED Y RANDOM AND A COUNT TO SEE IF USER ARRIVED IN USER SIMPLEX
        alter_dict = DataStructures.SortedDict()
        X_len = []
        Y_len = []
        for (ego, info) in global_sorted_dict
            egonet = info[3]
            alters = get_alters(egonet, ego)

            for alter in alters
                alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego) # get star alternetwork
                
                if length_alternetwork == len
                    alter_dict_len = get_alternetwork_n(alternetwork)

                    for (k, v) in alter_dict_len
                        try
                            push!(alter_dict[k], v)
                        catch e
                            alter_dict[k] = [v]
                        end
                    end
                end
            end
        end
        
        for (k, v) in alter_dict
            push!(X_len, k)
            push!(Y_len, mean(v))
        end

        push!(X, X_len)
        push!(Y, Y_len)
    end

    Plots.plot(X, Y, xlabel = "ordinal time", ylabel = "average number of nodes", xlim = (1, 30), label = ["Alternet Length 5" "Alternet Length 10" "Alternet Length 15" "Alternet Length 20" "Alternet Length 25" "Alternet Length 30"], left_margin = 10mm, bottom_margin = 10mm, legend =:bottomright, title = "Average number of nodes - Star", linewidth=0.5)
    Plots.savefig("$(dataset)_star_alternetwork_n.pdf")
end

function plot_alternetworks_time_until_user(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    X = []
    Y = []

    for len in [5, 10, 15, 20, 25, 30]
        X_len = []
        Y_len = []
        print(stdout, "Length: $len \r")
        flush(stdout)
        # NEED Y RANDOM AND A COUNT TO SEE IF USER ARRIVED IN USER SIMPLEX
        alter_dict = DataStructures.SortedDict()

        for (ego, info) in global_sorted_dict
            if info[5] == len
                egonet = info[1]
                alters = get_alters(egonet, ego)

                for alter in alters
                    alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego) # get star alternetwork
                    
                    try
                        push!(alter_dict[length_alternetwork], time_until_alter)
                    catch e
                        alter_dict[length_alternetwork] = [time_until_alter]
                    end
                    
                end
            end
        end

        for (k, v) in alter_dict
            push!(X_len, k)
            push!(Y_len, mean(v))
        end

        push!(X, X_len)
        push!(Y, Y_len)
    end

    Plots.plot(X, Y, xlabel = "degree of alter", ylabel = "time until alter arrives", label = ["Egonet Length 5" "Egonet Length 10" "Egonet Length 15" "Egonet Length 20" "Egonet Length 25" "Egonet Length 30"], xlim = (1, 30), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Time until alter arrives - Default", linewidth=0.5)
    Plots.savefig("$(dataset)_def_alternetwork_time_until_alter.pdf")

    #=
    for (k, v) in alter_dict
        push!(X, k)
        push!(Y, mean(v))
    end

    Plots.plot(X, Y, xlabel = "length of alternetwork", ylabel = "probability alter arrives in user simplex", left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "User vs Alter Simplicies - Default", linewidth=0.5)
    Plots.savefig("$(dataset)_def_alternetwork_arrive_user_simplex.pdf")
    =#


end

function plot_alternetworks_user_alter_first_simplex(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    X = 1:20
    Y_ran = []
    Y_actual = []

    alter_dict = DataStructures.SortedDict()
    for len in X
        print(stdout, "Length: $len \r")
        flush(stdout)

        sum = 0
        num_alters = 0
        average_prob = []

        for (ego, info) in global_sorted_dict
            egonet = info[1]
            alters = get_alters(egonet, ego)

            for alter in alters
                alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego)
                
                if length_alternetwork == len
                    # Did the alter arrive in a user simplex?
                    if ego in get_alternetwork_simplex(alternetwork, 1)
                        sum += 1
                    end

                    num_alters += 1
                    push!(average_prob, num_user_simplicies / length_alternetwork)
                end
                
            end
        end

        push!(Y_actual, sum / num_alters)
        push!(Y_ran, mean(average_prob))
    end

    Plots.plot(X, [Y_ran, Y_actual], xlabel = "length of alternetwork", ylabel = "probability alter arrives in user simplex", label = ["Random" "Actual"], xlim = (1, 20), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Probability alter arrives in user simplex - Default", linewidth=0.5)
    Plots.savefig("$(dataset)_def_alternetwork_arrive_user_simplex.pdf")

end

function plot_alternetworks_user_alter_simplicies(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 1:30
    
    Y_user = []
    Y_alter = []
    
    for l in X
        print(stdout, "Length: $l \r")
        flush(stdout)

        user_avg = []
        alter_avg = []

        for (ego, info) in global_sorted_dict
            egonet = info[1]
            alters = get_alters(egonet, ego)
            
            for alter in alters
                alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego)

                if length_alternetwork == l
                    push!(user_avg, num_user_simplicies)
                    push!(alter_avg, length_alternetwork - num_user_simplicies)
                end
            end
        end

        push!(Y_user, mean(user_avg))
        push!(Y_alter, mean(alter_avg))
    end
    
    Plots.plot(X, [Y_user, Y_alter], xlabel = "length of alternetwork", ylabel = "average number of simplicies", label = ["User Simplicies" "Alter Simplicies"], xlim=(1, 30), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Alternetwork - User vs Alter Simplicies - Default", linewidth=0.5)
    Plots.savefig("$(dataset)_def_alternetwork_num_simplicies.pdf")
end

function plot_alternetworks_predictions_star(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 1:30
    Y = []
    Y_random = []
    for l in X
        print(stdout, "Length: $l \r")
        flush(stdout)
        #println("Length: $length")
        #data_dict = DataStructures.SortedDict()
        sum = 0
        count = 0
        
        for (ego, info) in global_sorted_dict
            egonet = info[3]
            alters = get_alters(egonet, ego)

            for alter in alters
                alternetwork, length_alternetwork, num_user_simplicies, time_until_alter = get_alternetwork(egonet, alter, ego)
                if length_alternetwork == l
                    alternet_list = []

                    for (k, v) in alternetwork
                        for s in v
                            #println("Time: $k, Simplex: $s")
                            push!(alternet_list, s)
                        end
                    end
                    first_user_simplex = alternet_list[1]
                    #println("first: $first_user_simplex")
                    shuffled_simplicies = shuffle(alternet_list)
                    #data_dict[shuffled_simplicies] = first_user_simplex
                    pred_first_user_simplex = predict_first_size(shuffled_simplicies)
                    #println("pred: $pred_first_user_simplex")
                    if pred_first_user_simplex == first_user_simplex
                        #println("woo")
                        sum += 1
                    end

                    count += 1 
                end
            end
        end

        fraction_correct = sum / count
        push!(Y, fraction_correct)
        push!(Y_random, 1 / l)
    end

    Plots.plot(X, [Y, Y_random], xlabel = "length of alternet", ylabel = "fraction of correct guesses", label = ["Prediction" "Random Guessing"], xlim=(1, 30), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Fraction of correct guesses", linewidth=0.5)
    Plots.savefig("$(dataset)_alternetwork_predict_size.pdf")
end

function plot_user_alter_simplicies(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 5:100
    
    Y_user_def = []
    Y_alter_def = []

    Y_user_cont = []
    Y_alter_cont = []
    
    for l in X
        print(stdout, "Length: $l \r")
        flush(stdout)

        user_def = []
        alter_def = []

        user_cont = []
        alter_cont = []

        #def
        for (ego, info) in global_sorted_dict
            if info[5] == l
                egonet = info[1]
                
                user_simplicies, alter_simplicies = get_user_alter_simplicies(egonet, ego)

                num_user_simplicies = length(user_simplicies)
                num_alter_simplicies = length(alter_simplicies)

                push!(user_def, num_user_simplicies)
                push!(alter_def, num_alter_simplicies)

            end
            #cont
            if info[6] == l
                egonet = info[2] 
                
                user_simplicies, alter_simplicies = get_user_alter_simplicies(egonet, ego)

                num_user_simplicies = length(user_simplicies)
                num_alter_simplicies = length(alter_simplicies)

                push!(user_cont, num_user_simplicies)
                push!(alter_cont, num_alter_simplicies)
            end
        end

        push!(Y_user_def, mean(user_def))
        push!(Y_alter_def, mean(alter_def))
        
        push!(Y_user_cont, mean(user_cont))
        push!(Y_alter_cont, mean(alter_cont))
    end
    
    Plots.plot(X, [Y_user_def, Y_alter_def], xlabel = "length of default egonet", ylabel = "average number of simplicies", label = ["User Simplicies" "Alter Simplicies"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "User vs Alter Simplicies - Default", linewidth=0.5)
    Plots.savefig("$(dataset)_def_num_simplicies.pdf")

    Plots.plot(X, [Y_user_cont, Y_alter_cont], xlabel = "length of contracted egonet", ylabel = "average number of simplicies", label = ["User Simplicies" "Alter Simplicies"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "User vs Alter Simplicies - Contracted", linewidth=0.5)
    Plots.savefig("$(dataset)_cont_num_simplicies.pdf")
end

function plot_probability_first_simplex(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 5:100
    Y_user = []
    Y_alter = []
    for l in X
        print(stdout, "Length: $l \r")
        flush(stdout)
        sum_user = 0
        sum_alter = 0
        count = 0
        for (ego, info) in global_sorted_dict
            if info[6] == l
                egonet = info[2]
                first_user_simplex_list = []
                for (k, v) in egonet
                    first_user_simplex_list = v
                    break
                end
                if length(first_user_simplex_list) == 1
                    if ego in first_user_simplex_list[1]
                        sum_user += 1
                    else
                        sum_alter += 1
                    end
                    count += 1
                end
            end
        end

        fraction_user = sum_user / count
        fraction_alter = sum_alter / count
        push!(Y_user, fraction_user)
        #push!(Y_random, 1 / l)
        push!(Y_alter, fraction_alter)
    end

    Plots.plot(X, [Y_user, Y_alter], xlabel = "length of contracted egonet", ylabel = "probability for first simplex", label = ["User Simplex" "Alter Simplex"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Probabilty that first simplex is a user vs alter simplex", linewidth=0.5)
    Plots.savefig("$(dataset)_cont_first_simplex_user_alter.pdf")

end

function plot_predictions_until_user(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 1:50
    Y_novelty_size = []
    Y_random = []
    Y_correct = []
    Y_size = []
    Y_novelty = []
    Y_novelty_degree = []
    Y_degree = []
    Y_degree_novelty = []
    Y_degree_size = []
    Y_degree_novelty_size = []

    for l in X
        print(stdout, "Length: $l \r")
        flush(stdout)

        novelty_size_sum = 0
        size_sum = 0
        novelty_sum = 0
        novelty_degree_sum = 0
        count = 0
        lengths = []
        correct = []
        degree_sum = 0
        degree_novelty_sum = 0
        degree_size_sum = 0
        degree_novelty_size_sum = 0

        for (ego, info) in global_sorted_dict
            if info[5] == l
                egonet = info[1]
                first_user_simplex_list = []

                for (k, v) in egonet
                    first_user_simplex_list = v
                    break
                end

                #for (k, v) in egonet
                #    println("Time: $k, simp list: $v")
                #end

                if length(first_user_simplex_list) == 1
                    alter_simplicies, user_simplicies, first_user_simplex = get_simplicies_until_user(egonet, ego)
                    all_simplicies = get_all_simplicies(egonet)
                    pred_first_user_simplex_novelty_size = predict_first_until_user_novelty_size(alter_simplicies, user_simplicies, ego)
                    pred_first_user_simplex_novelty = predict_first_until_user_novelty(alter_simplicies, user_simplicies, ego)
                    pred_first_user_simplex_size = predict_first_size(user_simplicies)
                    pred_first_user_simplex_degree = predict_first_degree(user_simplicies, all_simplicies)
                    pred_first_user_simplex_degree_size = predict_first_until_user_degree_size(user_simplicies, all_simplicies)
                    # Out of the most novel, pick the highest degree
                    most_novel = get_top_novelty(alter_simplicies, user_simplicies, ego, 25)
                    pred_first_user_simplex_novelty_degree = predict_first_degree(most_novel, all_simplicies)
                    # Out of the highest degrees, pick the most novel
                    most_degree = get_top_degree(user_simplicies, all_simplicies, 25)
                    pred_first_user_simplex_degree_novelty = predict_first_until_user_novelty(alter_simplicies, most_degree, ego)
                    # Out of most novel and highest degree, pick smallest_pred_first_user_simplex
                    #println(user_simplicies)
                    most_degree_ = get_top_degree(user_simplicies, all_simplicies, 75)
                    #println(most_degree_)
                    most_degree_novelty = get_top_novelty(alter_simplicies, most_degree_, ego, 75)
                    pred_first_user_simplex_degree_novelty_size = predict_first_size(most_degree_novelty)

                    if pred_first_user_simplex_novelty_size == first_user_simplex
                        novelty_size_sum += 1
                    end

                    if pred_first_user_simplex_novelty == first_user_simplex
                        novelty_sum += 1
                    end

                    if pred_first_user_simplex_size == first_user_simplex
                        size_sum += 1
                    end

                    if pred_first_user_simplex_degree == first_user_simplex
                        degree_sum += 1
                    end

                    if pred_first_user_simplex_degree_size == first_user_simplex
                        degree_size_sum += 1
                    end

                    if pred_first_user_simplex_novelty_degree == first_user_simplex
                        novelty_degree_sum += 1
                    end

                    if pred_first_user_simplex_degree_novelty == first_user_simplex
                        degree_novelty_sum += 1
                    end

                    if pred_first_user_simplex_degree_novelty_size == first_user_simplex
                        degree_novelty_size_sum += 1
                    end

                    num_correct_options = get_num_occurences(egonet, first_user_simplex)

                    count += 1

                    push!(lengths, 1 / length(user_simplicies))
                    push!(correct, 1 / num_correct_options)
                end
            end
        end

        fraction_correct_novelty_size = novelty_size_sum / count
        fraction_correct_novelty = novelty_sum / count
        fraction_correct_size = size_sum / count
        fraction_correct_degree = degree_sum / count
        fraction_correct_degree_size = degree_size_sum / count
        fraction_correct_degree_novelty = degree_novelty_sum / count
        fraction_correct_novelty_degree = novelty_degree_sum / count
        fraction_correct_degree_novelty_size = degree_novelty_size_sum / count

        push!(Y_novelty_size, fraction_correct_novelty_size)
        push!(Y_novelty, fraction_correct_novelty)
        push!(Y_size, fraction_correct_size)
        push!(Y_degree, fraction_correct_degree)
        push!(Y_degree_size, fraction_correct_degree_size)
        push!(Y_novelty_degree, fraction_correct_novelty_degree)
        push!(Y_degree_novelty, fraction_correct_degree_novelty)
        push!(Y_degree_novelty_size, fraction_correct_degree_novelty_size)
        push!(Y_random, mean(lengths))
        push!(Y_correct, mean(correct))
    end

    Plots.plot(X, [Y_novelty_size, Y_random, Y_novelty, Y_size, Y_degree, Y_degree_size, Y_degree_novelty, Y_degree_novelty_size, Y_correct], xlabel = "length of default egonet", ylabel = "fraction of correct guesses", label = ["Novelty + Size" "Random Guessing" "Novelty" "Size" "Degree" "Degree + Size" "Degree + Novelty" "Degree + Novelty + Size" "Upper Bound"], #=xlim=(5, 100), =#left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Fraction of correct guesses - Default", linewidth=0.5)
    Plots.savefig("$(dataset)_predict_def_until_user.pdf")
end

function plot_predictions_star(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 1:30
    Y = []
    Y_random = []
    Y_size = []
    Y_degree_size = []
    Y_small_25 = []
    Y_small_50 = []
    Y_small_75 = []
    for l in X
        print(stdout, "Length: $l \r")
        flush(stdout)
        #println("Length: $length")
        #data_dict = DataStructures.SortedDict()
        sum = 0
        size_sum = 0
        small_sum_25 = 0
        small_sum_50 = 0
        small_sum_75 = 0
        degree_size_sum = 0
        count = 0
        lengths = []
        for (ego, info) in global_sorted_dict
            if info[7] == l
                egonet = info[3]
                first_user_simplex_list = []
                for (k, v) in egonet
                    #println("Time: $k, Simplex List: $v")
                    first_user_simplex_list = v
                    break
                end
                if length(first_user_simplex_list) == 1
                    user_simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
                    #println("First: $first_user_simplex")
                    #smallest_simplicies_25 = get_smallest_simplicies(user_simplicies, 25)
                    #smallest_simplicies_50 = get_smallest_simplicies(user_simplicies, 50)
                    #smallest_simplicies_75 = get_smallest_simplicies(user_simplicies, 75)
                    shuffled_simplicies = shuffle(user_simplicies)
                    pred_first_user_simplex_size = predict_first_size(shuffled_simplicies)
                    pred_first_user_simplex_degree_size = predict_first_until_user_degree_size(shuffled_simplicies, shuffled_simplicies)
                    #data_dict[shuffled_simplicies] = first_user_simplex
                    #pred_first_user_simplex = predict_first_size(shuffled_simplicies)
                    
                    #maybe only look at bottom 25% in size?
                    pred_first_user_simplex = predict_first_degree(shuffled_simplicies, user_simplicies)
                    #smallest_pred_first_user_simplex_25 = predict_first_degree(shuffle!(smallest_simplicies_25), user_simplicies)
                    #smallest_pred_first_user_simplex_50 = predict_first_degree(shuffle!(smallest_simplicies_50), user_simplicies)
                    #smallest_pred_first_user_simplex_75 = predict_first_degree(shuffle!(smallest_simplicies_75), user_simplicies)
                    #println("Prediction: $pred_first_user_simplex")
                    
                    if pred_first_user_simplex in first_user_simplex_list
                        sum += 1
                    end

                    if pred_first_user_simplex_degree_size in first_user_simplex_list
                        degree_size_sum += 1
                    end

                    if pred_first_user_simplex_size in first_user_simplex_list
                        size_sum += 1
                    end
                    #=
                    if smallest_pred_first_user_simplex_25 in first_user_simplex_list
                        small_sum_25 += 1
                    end
                    if smallest_pred_first_user_simplex_25 in first_user_simplex_list
                        small_sum_50 += 1
                    end
                    if smallest_pred_first_user_simplex_25 in first_user_simplex_list
                        small_sum_75 += 1
                    end
                    =#
                    count += 1
                    push!(lengths, 1 / length(user_simplicies))
                end
            end
        end

        fraction_correct = sum / count
        fraction_correct_degree_size = degree_size_sum / count
        fraction_correct_size = size_sum / count
        #fraction_correct_small = small_sum_25 / count
        #fraction_correct_small_50 = small_sum_50 / count
        #fraction_correct_small_75 = small_sum_75 / count
        push!(Y, fraction_correct)
        #push!(Y_small_25, fraction_correct_small)
        #push!(Y_small_50, fraction_correct_small_50)
        #push!(Y_small_75, fraction_correct_small_75)
        push!(Y_random, 1 / l)
        push!(Y_degree_size, fraction_correct_degree_size)
        push!(Y_size, fraction_correct_size)
        #push!(Y_random, mean(lengths))
    end

    Plots.plot(X, [Y, Y_random, Y_degree_size, Y_size], xlabel = "length of star egonet", ylabel = "fraction of correct guesses", label = ["Degree" "Random Guessing" "Degree + Size" "Size"], xlim=(1, 30), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Fraction of correct guesses", linewidth=0.5)
    Plots.savefig("$(dataset)_predict_star.pdf")

end

function plot_supersets(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    X = []
    Y = []

    for len in [5, 10, 15, 20, 25, 30]
        X_len = []
        Y_len = []
        print(stdout, "Length: $len \r")
        flush(stdout)
        superset_dict = DataStructures.SortedDict()

        for (ego, info) in global_sorted_dict
            if info[7] == len
                egonet = info[3]                
                simplicies = network_to_list(egonet)

                for i in 2:length(simplicies)
                    i_superset_j = 0

                    for j in 1:i-1
                        superset_check = [node for node in simplicies[j] if node in simplicies[i]]

                        if (length(superset_check) == length(simplicies[j]))
                            i_superset_j = 1
                            break
                        end
                    end

                    try
                        push!(superset_dict[i], i_superset_j)
                    catch e
                        superset_dict[i] = [i_superset_j]
                    end
                end

            end
        end

        superset_dict[1] = [0]

        for (k, v) in superset_dict
            push!(X_len, k)
            push!(Y_len, mean(v))
        end

        push!(X, X_len)
        push!(Y, Y_len)
    end

    Plots.plot(X, Y, xlabel = "length of star egonet", ylabel = "proportion of superset simplicies", label = ["Egonet Length 5" "Egonet Length 10" "Egonet Length 15" "Egonet Length 20" "Egonet Length 25" "Egonet Length 30"], xlim = (1, 30), left_margin = 10mm, bottom_margin = 10mm, legend =:bottomright, title = "Superset Simplicies - Star", linewidth=0.5)
    Plots.savefig("coauth-DBLP_star_supersets.pdf")
end

function plot_subsets(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    X = []
    Y = []

    for len in [5, 10, 15, 20, 25, 30]
        X_len = []
        Y_len = []
        print(stdout, "Length: $len \r")
        flush(stdout)
        subset_dict = DataStructures.SortedDict()

        for (ego, info) in global_sorted_dict
            if info[7] == len
                egonet = info[3]                
                simplicies = network_to_list(egonet)

                for i in 1:length(simplicies)
                    i_subset_j = 0

                    for j in i+1:length(simplicies)
                        subset_check = [node for node in simplicies[i] if node in simplicies[j]]

                        if (length(subset_check) == length(simplicies[i]))
                            i_subset_j = 1
                            break
                        end
                    end

                    try
                        push!(subset_dict[i], i_subset_j)
                    catch e
                        subset_dict[i] = [i_subset_j]
                    end
                end

            end
        end

        for (k, v) in subset_dict
            push!(X_len, k)
            push!(Y_len, mean(v))
        end

        push!(X, X_len)
        push!(Y, Y_len)
    end

    Plots.plot(X, Y, xlabel = "length of star egonet", ylabel = "proportion of subset simplicies", label = ["Egonet Length 5" "Egonet Length 10" "Egonet Length 15" "Egonet Length 20" "Egonet Length 25" "Egonet Length 30"], xlim = (1, 30), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Subset Simplicies - Star", linewidth=0.5)
    Plots.savefig("coauth-DBLP_star_subsets.pdf")
end

function plot_alternetwork_spread(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    X = []
    Y = []
    Y_rand = []

    spread_dict = DataStructures.SortedDict()
    random_spread_dict = DataStructures.SortedDict()

    for (ego, info) in global_sorted_dict
        print(stdout, "Ego: $ego \r")
        flush(stdout)

        if info[7] >= 10
            egonet = info[3]
            simplicies = network_to_list(egonet)
            alters = get_alters(egonet, ego)
            degree_dict = get_degree_dict(simplicies)
            #println("simplicies: $simplicies")
            #println()
            for alter in alters
                alternetwork_dict, count, num_user_simplicies, time_until_alter = get_alternetwork_shallow_copy(egonet, alter, ego)
                alternetwork = network_to_list(alternetwork_dict)
                #println("alter: $alter")
                if degree_dict[alter] > 1
                    i = 1
                    distances = []
                    while i < length(alternetwork)
                        push!(alternetwork[i], -1)
                        push!(alternetwork[i+1], -2)
                        #println("alternetwork[i]: $(alternetwork[i])")
                        #println("alternetwork[i+1]: $(alternetwork[i+1])")
                        distance = get_distances(simplicies, alternetwork[i], alternetwork[i+1])
                        #println("distance: $distance")
                        push!(distances, distance)

                        pop!(alternetwork[i])
                        pop!(alternetwork[i+1])
                        
                        i += 1
                    end
                    #println("distances: $distances")

                    if distances == []
                        #=
                        println("alters: $alters")
                        println("current alter: $alter")
                        println("simplicies: $simplicies")

                        for (k, v) in degree_dict
                            println("k: $k, v: $v")
                        end
                        =#
                        continue
                    end

                    avg_distance = mean(distances)

                    try
                        push!(spread_dict[degree_dict[alter]], avg_distance)
                    catch e
                        spread_dict[degree_dict[alter]] = [avg_distance]
                    end                    

                    #return
                end
            end

            # random baseline
            shuffled_simplicies = shuffle(simplicies)

            for alter in alters
                alternetwork_dict, count, num_user_simplicies, time_until_alter = get_alternetwork_shallow_copy(egonet, alter, ego)
                alternetwork = network_to_list(alternetwork_dict)
                #println("alter: $alter")
                if degree_dict[alter] > 1
                    i = 1
                    distances = []
                    while i < length(alternetwork)
                        push!(alternetwork[i], -1)
                        push!(alternetwork[i+1], -2)
                        #println("alternetwork[i]: $(alternetwork[i])")
                        #println("alternetwork[i+1]: $(alternetwork[i+1])")
                        distance = get_distances(shuffled_simplicies, alternetwork[i], alternetwork[i+1])
                        #println("distance: $distance")
                        push!(distances, distance)

                        pop!(alternetwork[i])
                        pop!(alternetwork[i+1])
                        
                        i += 1
                    end
                    #println("distances: $distances")

                    if distances == []
                        continue
                    end

                    avg_distance = mean(distances)

                    try
                        push!(random_spread_dict[degree_dict[alter]], avg_distance)
                    catch e
                        random_spread_dict[degree_dict[alter]] = [avg_distance]
                    end                    

                    #return
                end
            end
        end
        
        if ego == 10000
            break
        end
        
    end

    for (k, v) in spread_dict
        push!(X, k)
        push!(Y, mean(v))
    end

    for (k, v) in random_spread_dict
        push!(Y_rand, mean(v))
    end

    Plots.plot(X, [Y, Y_rand], #=seriestype = :scatter,=# xlabel = "degree of alter", ylabel = "avg distance between alternet simplicies", label = ["Actual" "Random"], left_margin = 10mm, bottom_margin = 10mm, xlim=(1, 100), legend = true, title = "Alternetwork Spread - Star", linewidth=0.5)
    Plots.savefig("coauth-DBLP_alternetwork_spread.pdf")
end

function num_egos(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    return length(global_sorted_dict)
end

function prepare_ML_set_length_data_star(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    #global_sorted_dict = load_sorted_dicts(dataset, true)
    sizes = []
    degrees = []
    occurences = []
    is_smallest = []
    avg_alternet_sizes = []
    is_first = []
    highest_max_mean = []
    count = 0

    for (ego, info) in global_sorted_dict
        print(stdout, "Count: $count \r")
        flush(stdout)
    
        if info[7] == len
            egonet = info[3]
            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            shuffle!(simplicies)

            for simplex in simplicies
                size = length(simplex)
                max_mean_degree = get_max_mean_degree(simplicies, simplex)
                num_occurences = get_num_occurences(egonet, simplex)
                simplex_sizes = get_sizes(simplicies)
                smallest_simplex_size = minimum(simplex_sizes)
                simplex_degrees = get_all_degrees_star(simplicies, simplicies)
                highest_degree = maximum(simplex_degrees)

                if size == smallest_simplex_size
                    push!(is_smallest, 1)
                else
                    push!(is_smallest, 0)
                end
                
                if max_mean_degree == highest_degree
                    push!(highest_max_mean, 1)
                else
                    push!(highest_max_mean, 0)
                end

                alt_sizes_to_avg = []
                for alter in simplex
                    if alter == ego
                        continue
                    end

                    alternet = get_alternetwork(egonet, alter, ego)[1]
                    alternet_list = network_to_list(alternet)
                    alternet_sizes = get_sizes(alternet_list)
                    avg_alternet_size = mean(alternet_sizes)

                    push!(alt_sizes_to_avg, avg_alternet_size)
                end

                push!(sizes, size)
                push!(degrees, max_mean_degree)
                push!(occurences, num_occurences)
                push!(avg_alternet_sizes, mean(alt_sizes_to_avg))

                if simplex == first_user_simplex
                    push!(is_first, 1)
                else
                    push!(is_first, 0)
                end
            end
        end
        count += 1
    end

    return sizes, degrees, occurences, avg_alternet_sizes, is_smallest, highest_max_mean, is_first
end

function prepare_ML_set_length_data_star_is_first(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a = []
    sizes_b = []
    occurences_a = []
    is_smallest_a = []
    highest_max_mean_a = []
    occurences_b = []
    is_smallest_b = []
    highest_max_mean_b = []
    sizes_dict = DataStructures.SortedDict()
    degrees_a = []
    degrees_b = []

    a_subset_b = []
    b_subset_a = []
    a_isolated_nodes = []
    b_isolated_nodes = []

    avg_sizes = []
    avg_degrees = []
    num_nodes = []
    # length_egonet = []

    degrees_dict = DataStructures.SortedDict()
    is_first = []
    count = 0
    in_count = 0

    for (ego, info) in global_sorted_dict
        print(stdout, "Count: $count \r")
        flush(stdout)
    
        if info[7] == len
            egonet = info[3]

            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            simplicies_without_first = filter(e->e!=first_user_simplex, simplicies)

            if length(simplicies_without_first) == 0
                continue
            end

            simplex = simplicies_without_first[rand(1:length(simplicies_without_first))]

            #simplex = sampled_simplex[1]                           
            a_and_b = [first_user_simplex, simplex]
            shuffle!(a_and_b)

            a = a_and_b[1]
            b = a_and_b[2]

            if a == b
                continue
            end

            #println("a: $a")
            #println("b: $b")
            #println()

            # set general data
            nodes = get_nodes(simplicies)
            push!(num_nodes, length(nodes))
            simplex_sizes = get_sizes(simplicies)
            avg_size = mean(simplex_sizes)
            push!(avg_sizes, avg_size)
            simplex_degrees = get_all_degrees_star(simplicies, simplicies, ego)
            #println("Simplex degrees: $simplex_degrees")
            avg_degree = mean(simplex_degrees)
            push!(avg_degrees, avg_degree)
            smallest_simplex_size = minimum(simplex_sizes)
            highest_degree = maximum(simplex_degrees)

            # set data for a
            push!(a, -1)
            a_isolated = get_isolated_nodes(simplicies, a)
            pop!(a)
            push!(a_isolated_nodes, length(a_isolated))
            size_a = length(a)
            max_mean_degree_a = get_max_mean_degree(simplicies, a, ego)
            num_occurences_a = get_num_occurences(egonet, a)

            if size_a == smallest_simplex_size
                push!(is_smallest_a, 1)
            else
                push!(is_smallest_a, 0)
            end
            
            if max_mean_degree_a == highest_degree
                push!(highest_max_mean_a, 1)
            else
                push!(highest_max_mean_a, 0)
            end

            push!(sizes_a, size_a)
            push!(degrees_a, max_mean_degree_a)
            push!(occurences_a, num_occurences_a)

            # set data for b
            push!(b, -1)
            b_isolated = get_isolated_nodes(simplicies, b)
            pop!(b)
            push!(b_isolated_nodes, length(b_isolated))
            size_b = length(b)
            max_mean_degree_b = get_max_mean_degree(simplicies, b, ego)
            num_occurences_b = get_num_occurences(egonet, b)

            if size_b == smallest_simplex_size
                push!(is_smallest_b, 1)
            else
                push!(is_smallest_b, 0)
            end
            
            if max_mean_degree_b == highest_degree
                push!(highest_max_mean_b, 1)
            else
                push!(highest_max_mean_b, 0)
            end

            push!(sizes_b, size_b)
            push!(degrees_b, max_mean_degree_b)
            push!(occurences_b, num_occurences_b)
            
            in_a = 1
            for node in a
                if !(node in b)
                    in_a = 0
                    break
                end
            end

            push!(a_subset_b, in_a)

            in_b = 1
            for node in b
                if !(node in a)
                    in_b = 0
                    break
                end
            end

            push!(b_subset_a, in_b)

            #=
            if a in combinations(b)
                push!(a_subset_b, 1)
            else
                push!(a_subset_b, 0)
            end

            if b in combinations(a)
                push!(b_subset_a, 1)
            else
                push!(b_subset_a, 0)
            end
            =#

            push!(a, -1)
            push!(b, -1)
            #println("Simplicies: $simplicies")

            if a == first_user_simplex
                push!(is_first, 1)
            elseif b == first_user_simplex
                push!(is_first, 0)
            end

            pop!(a)
            pop!(b)

            in_count += 1
            #println()
        end
        count += 1
        #=
        if in_count == 15
            println("a_subset_b: $a_subset_b")
            println("b_subset_a: $b_subset_a")
            #println()
            println("a_isolated_nodes: $a_isolated_nodes")
            println("b_isolated_nodes: $b_isolated_nodes")
            println("avg_sizes: $avg_sizes")
            println("avg_degrees: $avg_degrees")
            println("num_nodes: $num_nodes")

            println(is_first)
            return
        end
        =#
    end

    return sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, is_first
end

function prepare_ML_set_length_data_star_is_first_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a = []
    sizes_b = []
    occurences_a = []
    is_smallest_a = []
    highest_max_mean_a = []
    occurences_b = []
    is_smallest_b = []
    highest_max_mean_b = []
    sizes_dict = DataStructures.SortedDict()
    degrees_a = []
    degrees_b = []
    lengths = []

    a_subset_b = []
    b_subset_a = []
    a_isolated_nodes = []
    b_isolated_nodes = []

    avg_sizes = []
    avg_degrees = []
    num_nodes = []
    # length_egonet = []

    degrees_dict = DataStructures.SortedDict()
    is_first = []
    count = 0
    in_count = 0

    for (ego, info) in global_sorted_dict
        print(stdout, "Count: $count \r")
        flush(stdout)
    
        if 10 <= info[7] <= 20
            egonet = info[3]

            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            simplicies_without_first = filter(e->e!=first_user_simplex, simplicies)

            if length(simplicies_without_first) == 0
                continue
            end

            simplex = simplicies_without_first[rand(1:length(simplicies_without_first))]

            #simplex = sampled_simplex[1]                           
            a_and_b = [first_user_simplex, simplex]
            shuffle!(a_and_b)

            a = a_and_b[1]
            b = a_and_b[2]

            if a == b
                continue
            end

            #println("a: $a")
            #println("b: $b")
            #println()

            # set general data
            nodes = get_nodes(simplicies)
            push!(num_nodes, length(nodes))
            simplex_sizes = get_sizes(simplicies)
            avg_size = mean(simplex_sizes)
            push!(avg_sizes, avg_size)
            simplex_degrees = get_all_degrees_star(simplicies, simplicies, ego)
            #println("Simplex degrees: $simplex_degrees")
            avg_degree = mean(simplex_degrees)
            push!(avg_degrees, avg_degree)
            smallest_simplex_size = minimum(simplex_sizes)
            highest_degree = maximum(simplex_degrees)
            push!(lengths, info[7])

            # set data for a
            push!(a, -1)
            a_isolated = get_isolated_nodes(simplicies, a)
            pop!(a)
            push!(a_isolated_nodes, length(a_isolated))
            size_a = length(a)
            max_mean_degree_a = get_max_mean_degree(simplicies, a, ego)
            num_occurences_a = get_num_occurences(egonet, a)

            if size_a == smallest_simplex_size
                push!(is_smallest_a, 1)
            else
                push!(is_smallest_a, 0)
            end
            
            if max_mean_degree_a == highest_degree
                push!(highest_max_mean_a, 1)
            else
                push!(highest_max_mean_a, 0)
            end

            push!(sizes_a, size_a)
            push!(degrees_a, max_mean_degree_a)
            push!(occurences_a, num_occurences_a)

            # set data for b
            push!(b, -1)
            b_isolated = get_isolated_nodes(simplicies, b)
            pop!(b)
            push!(b_isolated_nodes, length(b_isolated))
            size_b = length(b)
            max_mean_degree_b = get_max_mean_degree(simplicies, b, ego)
            num_occurences_b = get_num_occurences(egonet, b)

            if size_b == smallest_simplex_size
                push!(is_smallest_b, 1)
            else
                push!(is_smallest_b, 0)
            end
            
            if max_mean_degree_b == highest_degree
                push!(highest_max_mean_b, 1)
            else
                push!(highest_max_mean_b, 0)
            end

            push!(sizes_b, size_b)
            push!(degrees_b, max_mean_degree_b)
            push!(occurences_b, num_occurences_b)
            
            in_a = 1
            for node in a
                if !(node in b)
                    in_a = 0
                    break
                end
            end

            push!(a_subset_b, in_a)

            in_b = 1
            for node in b
                if !(node in a)
                    in_b = 0
                    break
                end
            end

            push!(b_subset_a, in_b)

            #=
            if a in combinations(b)
                push!(a_subset_b, 1)
            else
                push!(a_subset_b, 0)
            end

            if b in combinations(a)
                push!(b_subset_a, 1)
            else
                push!(b_subset_a, 0)
            end
            =#

            push!(a, -1)
            push!(b, -1)
            #println("Simplicies: $simplicies")

            if a == first_user_simplex
                push!(is_first, 1)
            elseif b == first_user_simplex
                push!(is_first, 0)
            end

            pop!(a)
            pop!(b)

            in_count += 1
            #println()
        end
        count += 1
        #=
        if in_count == 15
            println("a_subset_b: $a_subset_b")
            println("b_subset_a: $b_subset_a")
            #println()
            println("a_isolated_nodes: $a_isolated_nodes")
            println("b_isolated_nodes: $b_isolated_nodes")
            println("avg_sizes: $avg_sizes")
            println("avg_degrees: $avg_degrees")
            println("num_nodes: $num_nodes")

            println(is_first)
            return
        end
        =#
    end

    return sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, lengths, is_first
end

function prepare_ML_set_length_data_star_a_before_b(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a = []
    sizes_b = []
    occurences_a = []
    is_smallest_a = []
    highest_max_mean_a = []
    occurences_b = []
    is_smallest_b = []
    highest_max_mean_b = []
    sizes_dict = DataStructures.SortedDict()
    degrees_a = []
    degrees_b = []

    a_subset_b = []
    b_subset_a = []
    a_isolated_nodes = []
    b_isolated_nodes = []

    avg_sizes = []
    avg_degrees = []
    num_nodes = []
    # length_egonet = []

    degrees_dict = DataStructures.SortedDict()
    #intersection = []
    a_before_b = []
    count = 0
    in_count = 0

    for (ego, info) in global_sorted_dict
        print(stdout, "Count: $count \r")
        flush(stdout)
    
        if info[7] == len
            egonet = info[3]

            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            simplicies_without_first = filter(e->e!=first_user_simplex, simplicies)
            #println("Simplicies: $simplicies")
            if length(simplicies_without_first) == 0
                continue
            end

            sampled_simplicies =
                simplicies[StatsBase.sample(1:length(simplicies),
                                            2, replace=false)]

            a = sampled_simplicies[1]
            b = sampled_simplicies[2]

            #println("a: $a")
            #println("b: $b")

            if (a == b)
                #println("Skipped!")
                #println()
                continue
            end

            # set general data
            nodes = get_nodes(simplicies)
            push!(num_nodes, length(nodes))
            simplex_sizes = get_sizes(simplicies)
            avg_size = mean(simplex_sizes)
            push!(avg_sizes, avg_size)
            simplex_degrees = get_all_degrees_star(simplicies, simplicies, ego)
            #println("Simplex degrees: $simplex_degrees")
            avg_degree = mean(simplex_degrees)
            push!(avg_degrees, avg_degree)
            smallest_simplex_size = minimum(simplex_sizes)
            highest_degree = maximum(simplex_degrees)

            # set data for a
            push!(a, -1)
            a_isolated = get_isolated_nodes(simplicies, a)
            pop!(a)
            push!(a_isolated_nodes, length(a_isolated))
            size_a = length(a)
            max_mean_degree_a = get_max_mean_degree(simplicies, a, ego)
            num_occurences_a = get_num_occurences(egonet, a)

            if size_a == smallest_simplex_size
                push!(is_smallest_a, 1)
            else
                push!(is_smallest_a, 0)
            end
            
            if max_mean_degree_a == highest_degree
                push!(highest_max_mean_a, 1)
            else
                push!(highest_max_mean_a, 0)
            end

            push!(sizes_a, size_a)
            push!(degrees_a, max_mean_degree_a)
            push!(occurences_a, num_occurences_a)

            # set data for b
            push!(b, -1)
            b_isolated = get_isolated_nodes(simplicies, b)
            pop!(b)
            push!(b_isolated_nodes, length(b_isolated))
            size_b = length(b)
            max_mean_degree_b = get_max_mean_degree(simplicies, b, ego)
            num_occurences_b = get_num_occurences(egonet, b)

            if size_b == smallest_simplex_size
                push!(is_smallest_b, 1)
            else
                push!(is_smallest_b, 0)
            end
            
            if max_mean_degree_b == highest_degree
                push!(highest_max_mean_b, 1)
            else
                push!(highest_max_mean_b, 0)
            end

            push!(sizes_b, size_b)
            push!(degrees_b, max_mean_degree_b)
            push!(occurences_b, num_occurences_b)
            
            in_a = 1
            for node in a
                if !(node in b)
                    in_a = 0
                    break
                end
            end

            push!(a_subset_b, in_a)

            in_b = 1
            for node in b
                if !(node in a)
                    in_b = 0
                    break
                end
            end

            push!(b_subset_a, in_b)

            #push!(intersection, length(intersect(a, b)))
            push!(a, -1)
            push!(b, -1)
            #println("Simplicies: $simplicies")
            for s in simplicies
                if s == a
                    push!(a_before_b, 1)
                    break
                elseif s == b
                    push!(a_before_b, 0)
                    break
                end
            end

            pop!(a)
            pop!(b)
            #println()
            in_count += 1
        end

        count += 1
        #if in_count == 5
        #    println(a_before_b)
        #    return
        #end

    end

    return sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, a_before_b
end

function prepare_ML_set_length_data_star_a_before_b_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a = []
    sizes_b = []
    occurences_a = []
    is_smallest_a = []
    highest_max_mean_a = []
    occurences_b = []
    is_smallest_b = []
    highest_max_mean_b = []
    sizes_dict = DataStructures.SortedDict()
    degrees_a = []
    degrees_b = []
    lengths = []

    a_subset_b = []
    b_subset_a = []
    a_isolated_nodes = []
    b_isolated_nodes = []

    avg_sizes = []
    avg_degrees = []
    num_nodes = []

    degrees_dict = DataStructures.SortedDict()
    intersection = []
    a_before_b = []
    count = 0
    in_count = 0

    for (ego, info) in global_sorted_dict
        print(stdout, "Count: $count \r")
        flush(stdout)
    
        if 10 <= info[7] <= 20
            egonet = info[3]

            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            simplicies_without_first = filter(e->e!=first_user_simplex, simplicies)
            #println("Simplicies: $simplicies")
            if length(simplicies_without_first) == 0
                continue
            end

            sampled_simplicies =
                simplicies[StatsBase.sample(1:length(simplicies),
                                            2, replace=false)]

            a = sampled_simplicies[1]
            b = sampled_simplicies[2]

            #println("a: $a")
            #println("b: $b")

            if (a == b)
                #println("Skipped!")
                #println()
                continue
            end

            # set general data
            nodes = get_nodes(simplicies)
            push!(num_nodes, length(nodes))
            simplex_sizes = get_sizes(simplicies)
            avg_size = mean(simplex_sizes)
            push!(avg_sizes, avg_size)
            simplex_degrees = get_all_degrees_star(simplicies, simplicies, ego)
            #println("Simplex degrees: $simplex_degrees")
            avg_degree = mean(simplex_degrees)
            push!(avg_degrees, avg_degree)
            smallest_simplex_size = minimum(simplex_sizes)
            highest_degree = maximum(simplex_degrees)
            push!(lengths, info[7])
            push!(intersection, length(intersect(a, b)))

            # set data for a
            push!(a, -1)
            a_isolated = get_isolated_nodes(simplicies, a)
            pop!(a)
            push!(a_isolated_nodes, length(a_isolated))
            size_a = length(a)
            max_mean_degree_a = get_max_mean_degree(simplicies, a, ego)
            num_occurences_a = get_num_occurences(egonet, a)

            if size_a == smallest_simplex_size
                push!(is_smallest_a, 1)
            else
                push!(is_smallest_a, 0)
            end
            
            if max_mean_degree_a == highest_degree
                push!(highest_max_mean_a, 1)
            else
                push!(highest_max_mean_a, 0)
            end

            push!(sizes_a, size_a)
            push!(degrees_a, max_mean_degree_a)
            push!(occurences_a, num_occurences_a)

            # set data for b
            push!(b, -1)
            b_isolated = get_isolated_nodes(simplicies, b)
            pop!(b)
            push!(b_isolated_nodes, length(b_isolated))
            size_b = length(b)
            max_mean_degree_b = get_max_mean_degree(simplicies, b, ego)
            num_occurences_b = get_num_occurences(egonet, b)

            if size_b == smallest_simplex_size
                push!(is_smallest_b, 1)
            else
                push!(is_smallest_b, 0)
            end
            
            if max_mean_degree_b == highest_degree
                push!(highest_max_mean_b, 1)
            else
                push!(highest_max_mean_b, 0)
            end

            push!(sizes_b, size_b)
            push!(degrees_b, max_mean_degree_b)
            push!(occurences_b, num_occurences_b)
            
            in_a = 1
            for node in a
                if !(node in b)
                    in_a = 0
                    break
                end
            end

            push!(a_subset_b, in_a)

            in_b = 1
            for node in b
                if !(node in a)
                    in_b = 0
                    break
                end
            end

            push!(b_subset_a, in_b)

            push!(a, -1)
            push!(b, -1)
            #println("Simplicies: $simplicies")
            for s in simplicies
                if s == a
                    push!(a_before_b, 1)
                    break
                elseif s == b
                    push!(a_before_b, 0)
                    break
                end
            end

            pop!(a)
            pop!(b)
            #println()
            in_count += 1
        end

        count += 1
        #if in_count == 5
        #    println(a_before_b)
        #    return
        #end

    end

    return sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, lengths, intersection, a_before_b
end

function prepare_ML_set_length_data_star_adjacent(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a = []
    sizes_b = []
    occurences_a = []
    is_smallest_a = []
    highest_max_mean_a = []
    occurences_b = []
    is_smallest_b = []
    highest_max_mean_b = []
    sizes_dict = DataStructures.SortedDict()
    degrees_a = []
    degrees_b = []
    degrees_dict = DataStructures.SortedDict()
    a_adjacent_b = []
    intersection = []
    count = 0
    in_count = 0

    for (ego, info) in global_sorted_dict
        print(stdout, "Count: $count \r")
        flush(stdout)
    
        if info[7] == len
            egonet = info[3]

            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            simplicies_without_first = filter(e->e!=first_user_simplex, simplicies)
            #println("Simplicies: $simplicies")
            if length(simplicies_without_first) == 0
                continue
            end

            # get all possible pairs
            # split into two arrays: one of all adjacent pairs, and one of all non-adjacent pairs
            # sample from these two

            pairs = combinations(simplicies, 2)
            
            adjacent_pairs = []

            for i in 1:(length(simplicies) - 1)
                push!(adjacent_pairs, [simplicies[i], simplicies[i+1]])
            end

            #adjacent_pairs = [p for p in pairs if (get_distances(simplicies, p[1], p[2]) == 1)]
            #println("Adjacent pairs: $adjacent_pairs")
            #println("Length of  adjacent pairs: $(length(adjacent_pairs))")

            non_adjacent_pairs = []

            for i in 1:length(simplicies)
                for j in i+2:length(simplicies)
                    push!(non_adjacent_pairs, [simplicies[i], simplicies[j]])
                end
            end

            #println("Non adjacent pairs: $non_adjacent_pairs")
            #println("Length of non adjacent pairs: $(length(non_adjacent_pairs))")
            sampled_adjacent = adjacent_pairs[rand(1:length(adjacent_pairs))]
            sampled_non_adjacent = non_adjacent_pairs[rand(1:length(non_adjacent_pairs))]
            #println("Sampled adjacent pair: $sampled_adjacent")
            #println("Sampled non-adjacent pair: $sampled_non_adjacent")

            samples = [sampled_adjacent, sampled_non_adjacent]

            final_sampled_pair = samples[rand(1:length(samples))]
            #println("Final sampled pair: $final_sampled_pair")
            a = final_sampled_pair[1]
            b = final_sampled_pair[2]
            #push!(-1, a)
            #push!(-1, b)
            #println("A: $a")
            #println("B: $b")

            if (a == b)
                #println("Skipped!")
                #println()
                continue
            end

            # set data for a
            size_a = length(a)
            max_mean_degree_a = get_max_mean_degree(simplicies, a, ego)
            num_occurences_a = get_num_occurences(egonet, a)
            simplex_sizes = get_sizes(simplicies)
            smallest_simplex_size = minimum(simplex_sizes)
            simplex_degrees = get_all_degrees_star(simplicies, simplicies, ego)
            highest_degree = maximum(simplex_degrees)

            if size_a == smallest_simplex_size
                push!(is_smallest_a, 1)
            else
                push!(is_smallest_a, 0)
            end
            
            if max_mean_degree_a == highest_degree
                push!(highest_max_mean_a, 1)
            else
                push!(highest_max_mean_a, 0)
            end

            # set data for b
            size_b = length(b)
            max_mean_degree_b = get_max_mean_degree(simplicies, b, ego)
            num_occurences_b = get_num_occurences(egonet, b)

            if size_b == smallest_simplex_size
                push!(is_smallest_b, 1)
            else
                push!(is_smallest_b, 0)
            end
            
            if max_mean_degree_b == highest_degree
                push!(highest_max_mean_b, 1)
            else
                push!(highest_max_mean_b, 0)
            end

            push!(sizes_a, size_a)
            push!(degrees_a, max_mean_degree_a)
            push!(occurences_a, num_occurences_a)

            push!(sizes_b, size_b)
            push!(degrees_b, max_mean_degree_b)
            push!(occurences_b, num_occurences_b)

            push!(intersection, length(intersect(a, b)) - 1)

            push!(a, -1)
            push!(b, -1)
            #println("Simplicies: $simplicies")
            if get_distances(simplicies, a, b) == 1
                push!(a_adjacent_b, 1)
            else
                push!(a_adjacent_b, 0)
            end

            pop!(a)
            pop!(b)
            #println()
            in_count += 1
        end
        count += 1
        
        #if in_count == 5
        #    println(a_adjacent_b)
        #    return
        #end
        
    end

    return sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, sizes_b, degrees_b, occurences_b, is_smallest_b, highest_max_mean_b, intersection, a_adjacent_b
end

function prepare_ML_set_length_data_star_simplicies(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    spread_index_a = []
    spread_index_b = []

    smallest_first_a = []
    smallest_first_b = []

    a_first_subset = []
    b_first_subset = []

    a_last_superset = []
    b_last_superset = []

    size_rate_of_change_a = []
    size_rate_of_change_b = []

    a_is_sorted = []

    egonets = []

    println("Preparing data...")
    for (ego, info) in global_sorted_dict
        if len == info[7]
            egonet = info[3]
            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            push!(egonets, simplicies)
        end
    end
    println("Finished preparing data!")

    for i in 1:2:length(egonets)-1
        print(stdout, "Count: $i \r")
        flush(stdout)
        
        egonet_pair = [egonets[i], egonets[i+1]]
        shuffle!(egonet_pair)

        a = egonet_pair[1]
        b = egonet_pair[2]

        a_and_b = [a, b]
        index = rand([1, 2])
        shuffle!(a_and_b[index])

        #println("a: $a")
        #println("b: $b")

        if index == 1
            #println("b is sorted!")
            push!(a_is_sorted, 0)
        else
            #println("a is sorted!")
            push!(a_is_sorted, 1)
        end

        # set general data a
        simplex_sizes_a = get_sizes(a)
        avg_size_a = mean(simplex_sizes_a)
        simplex_degrees_a = get_all_degrees_star(a, a)
        smallest_simplex_size_a = minimum(simplex_sizes_a)
        highest_degree_a = maximum(simplex_degrees_a)

        # set general data b
        simplex_sizes_b = get_sizes(b)
        avg_size_b = mean(simplex_sizes_b)
        simplex_degrees_b = get_all_degrees_star(b, b)
        smallest_simplex_size_b = minimum(simplex_sizes_b)
        highest_degree_b = maximum(simplex_degrees_b)

        # number of isolated nodes in first user simplex

        # is smallest simplex first
        if length(a[1]) == smallest_simplex_size_a
            push!(smallest_first_a, 1)
        else
            push!(smallest_first_a, 0)
        end

        if length(b[1]) == smallest_simplex_size_b
            push!(smallest_first_b, 1)
        else
            push!(smallest_first_b, 0)
        end

        # the first simplex is a subset of how many future sets?
        push!(a_first_subset, count_subsets(a, a[1])) 
        push!(b_first_subset, count_subsets(b, b[1]))

        # the last simplex is a superset of how many previous sets?
        push!(a_last_superset, count_supersets(a, a[length(a)])) 
        push!(b_last_superset, count_supersets(b, b[length(b)]))

        # spread index
        intersections_a = []
        intersections_b = []

        for j in 1:length(a)-1
            in_a = length(intersect(a[j], a[j+1]))
            push!(intersections_a, in_a)

            in_b = length(intersect(b[j], b[j+1]))
            push!(intersections_b, in_b)
        end

        push!(spread_index_a, mean(intersections_a) / avg_size_a)
        push!(spread_index_b, mean(intersections_b) / avg_size_b)

        # calculate size rate of change
        push!(size_rate_of_change_a, (length(a[length(a)]) - length(a[1]))/length(a))
        push!(size_rate_of_change_b, (length(b[length(b)]) - length(b[1]))/length(b))

        #println()
    end
    #=
    println("spread_index_a: $spread_index_a")
    println("spread_index_b: $spread_index_b")
    println()
    println("smallest_first_a: $smallest_first_a")
    println("smallest_first_b: $smallest_first_b")
    println()
    println("a_first_subset: $a_first_subset")
    println("b_first_subset: $b_first_subset")
    println()
    println("a_last_superset: $a_last_superset")
    println("b_last_superset: $b_last_superset")
    println()
    println("size_rate_of_change_a: $size_rate_of_change_a")
    println("size_rate_of_change_b: $size_rate_of_change_b")
    println()
    println("a_is_sorted: $a_is_sorted")
    =#
    return spread_index_a, spread_index_b, smallest_first_a, smallest_first_b, a_first_subset, b_first_subset, 
        a_last_superset, b_last_superset, size_rate_of_change_a, size_rate_of_change_b, a_is_sorted
end

function prepare_ML_set_length_data_star_simplicies_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    lengths = []
    spread_index = []
    avg_alternet_spread_index = []
    smallest_first = []
    first_subset = []
    last_superset = []
    size_rate_of_change = []
    alternet_spread = []
    early_size_index = []
    early_deg_index = []

    is_sorted = []

    egonets = DataStructures.SortedDict()

    println("Preparing data...")
    for (ego, info) in global_sorted_dict
        #=
        if 100 <= info[7] <= 400
            egonet = info[3]
            simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
            egonets[ego] = simplicies
        end
        =#
        egonet = info[2]
        simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
        egonets[ego] = simplicies
    end
    println("Finished preparing data! Num egos: $(length(egonets))")

    count = 1

    for (ego, simplicies) in egonets
        print(stdout, "Count: $count \r")
        flush(stdout)
        #println("ego: $ego")
        a = deepcopy(simplicies)
        #println("a: $a")

        index = rand([0, 1])

        if index == 0
            shuffle!(a)
            #println("shuffle!")
            push!(is_sorted, 0)
        else
            push!(is_sorted, 1)
        end
        #println("a: $a")

        # average alternet spread
        alters = get_alters_simplicies(a, ego)
        degree_dict = get_degree_dict(a)
        avg_distances_alters = []
        alter_spread_index = []
        for alter in alters
            alternetwork, count_, num_user_simplicies, time_until_alter = get_alternetwork_shallow_copy_simplicies(a, alter, ego)
            #println("alter: $alter")
            if degree_dict[alter] > 1
                i = 1
                distances = []
                intersections_alter = []
                while i < length(alternetwork)
                    push!(alternetwork[i], -1)
                    push!(alternetwork[i+1], -2)
                    #println("alternetwork[i]: $(alternetwork[i])")
                    #println("alternetwork[i+1]: $(alternetwork[i+1])")
                    distance = get_distances(a, alternetwork[i], alternetwork[i+1])
                    #println("distance: $distance")
                    push!(distances, distance)

                    pop!(alternetwork[i])
                    pop!(alternetwork[i+1])
                    
                    in_a = length(intersect(alternetwork[i], alternetwork[i+1]))
                    push!(intersections_alter, in_a)

                    i += 1
                end

                avg_size_alter = mean([length(s) for s in alternetwork])

                if distances == []
                    continue
                end

                push!(alter_spread_index, mean(intersections_alter) / avg_size_alter)

                avg_distance = mean(distances)
                push!(avg_distances_alters, avg_distance)            
            end
        end
        if avg_distances_alters == []
            pop!(is_sorted)
            continue
        else
            push!(alternet_spread, mean(avg_distances_alters))
            push!(avg_alternet_spread_index, mean(alter_spread_index))
        end

        # set general data a
        simplex_sizes_a = get_sizes(a)
        avg_size_a = mean(simplex_sizes_a)
        simplex_degrees_a = get_all_degrees_star(a, a)
        avg_deg_a = mean(simplex_degrees_a)
        smallest_simplex_size_a = minimum(simplex_sizes_a)
        highest_degree_a = maximum(simplex_degrees_a)
        push!(lengths, length(a))

        # avg size/degree in first 20%
        index_20 = Int64(round(length(a)*0.2))
        degs = []
        sizes = []
        degree_dict = get_degree_dict(a)
        for i in 1:index_20
            size_simp = length(a[i])
            deg_simp = simplex_degrees_a[i]
            push!(sizes, size_simp)
            push!(degs, deg_simp)
        end
        avg_size = mean(sizes)
        avg_deg = mean(degs)
        push!(early_size_index, avg_size / avg_size_a)
        push!(early_deg_index, avg_deg / avg_deg_a)
        #println("a: $a")
        #println("index_20: $index_20")
        #println("early_size_index: $early_size_index")
        #println("early_deg_index: $early_deg_index")
        #return

        # is smallest simplex first
        if length(a[1]) == smallest_simplex_size_a
            push!(smallest_first, 1)
        else
            push!(smallest_first, 0)
        end

        # the first simplex is a subset of how many future sets?
        push!(first_subset, count_subsets(a, a[1])) 

        # the last simplex is a superset of how many previous sets?
        push!(last_superset, count_supersets(a, a[length(a)])) 

        # spread index
        intersections_a = []
        for j in 1:length(a)-1
            in_a = length(intersect(a[j], a[j+1]))
            push!(intersections_a, in_a)
        end
        push!(spread_index, mean(intersections_a) / avg_size_a)

        # calculate size rate of change
        push!(size_rate_of_change, (length(a[length(a)]) - length(a[1]))/length(a))

        #=
        println()
        if count == 5
            println("alternet_spread: $alternet_spread")
            println("spread_index: $spread_index")
            println("avg_alternet_spread_index: $avg_alternet_spread_index")
            println("is_sorted: $is_sorted")
            return
        end
        =#
        count += 1
    end

    return spread_index, smallest_first, first_subset, 
        last_superset, size_rate_of_change, lengths, alternet_spread, early_size_index, early_deg_index, 
        avg_alternet_spread_index, is_sorted
end

function prepare_ML_set_length_data_def_cont_simplicies_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    lengths = []
    spread_index = []
    avg_alternet_spread_index = []
    smallest_first = []
    first_subset = []
    last_superset = []
    size_rate_of_change = []
    alternet_spread = []
    early_size_index = []
    early_deg_index = []
    time_until_users = []
    size_first_user_simplex = []
    novelty_first_user_simplex = []

    is_sorted = []

    egonets = DataStructures.SortedDict()

    println("Preparing data...")
    for (ego, info) in global_sorted_dict
        if 10 <= info[7] <= 100
            egonet = info[1]
            simplicies = network_to_list(egonet)
            egonets[ego] = simplicies
        end
    end
    println("Finished preparing data! Num egos: $(length(egonets))")

    count = 1

    for (ego, simplicies) in egonets
        print(stdout, "Count: $count \r")
        flush(stdout)
        #println("ego: $ego")
        a = deepcopy(simplicies)
        #println("a: $a")

        index = rand([0, 1])

        if index == 0
            shuffle!(a)
            #println("shuffle!")
            push!(is_sorted, 0)
        else
            push!(is_sorted, 1)
        end
        #println("a: $a")

        # average alternet spread
        alters = get_alters_simplicies(a, ego)
        degree_dict = get_degree_dict(a)
        avg_distances_alters = []
        alter_spread_index = []
        for alter in alters
            alternetwork, count_, num_user_simplicies, time_until_alter = get_alternetwork_shallow_copy_simplicies(a, alter, ego)
            #println("alter: $alter")
            if degree_dict[alter] > 1
                i = 1
                distances = []
                intersections_alter = []
                while i < length(alternetwork)
                    push!(alternetwork[i], -1)
                    push!(alternetwork[i+1], -2)
                    #println("alternetwork[i]: $(alternetwork[i])")
                    #println("alternetwork[i+1]: $(alternetwork[i+1])")
                    distance = get_distances(a, alternetwork[i], alternetwork[i+1])
                    #println("distance: $distance")
                    push!(distances, distance)

                    pop!(alternetwork[i])
                    pop!(alternetwork[i+1])
                    
                    in_a = length(intersect(alternetwork[i], alternetwork[i+1]))
                    push!(intersections_alter, in_a)

                    i += 1
                end

                avg_size_alter = mean([length(s) for s in alternetwork])

                if distances == []
                    continue
                end

                push!(alter_spread_index, mean(intersections_alter) / avg_size_alter)

                avg_distance = mean(distances)
                push!(avg_distances_alters, avg_distance)            
            end
        end
        if avg_distances_alters == []
            pop!(is_sorted)
            continue
        else
            push!(alternet_spread, mean(avg_distances_alters))
            push!(avg_alternet_spread_index, mean(alter_spread_index))
        end

        # set general data a
        simplex_sizes_a = get_sizes(a)
        avg_size_a = mean(simplex_sizes_a)
        #simplex_degrees_a = get_all_degrees_star(a, a)
        #avg_deg_a = mean(simplex_degrees_a)
        smallest_simplex_size_a = minimum(simplex_sizes_a)
        #highest_degree_a = maximum(simplex_degrees_a)
        push!(lengths, length(a))
        #=
        # avg size/degree in first 20%
        index_20 = Int64(round(length(a)*0.2))
        degs = []
        sizes = []
        degree_dict = get_degree_dict(a)
        for i in 1:index_20
            size_simp = length(a[i])
            deg_simp = simplex_degrees_a[i]
            push!(sizes, size_simp)
            push!(degs, deg_simp)
        end
        avg_size = mean(sizes)
        avg_deg = mean(degs)
        push!(early_size_index, avg_size / avg_size_a)
        push!(early_deg_index, avg_deg / avg_deg_a)
        =#
        #println("a: $a")
        #println("index_20: $index_20")
        #println("early_size_index: $early_size_index")
        #println("early_deg_index: $early_deg_index")
        #return

        # is smallest simplex first
        if length(a[1]) == smallest_simplex_size_a
            push!(smallest_first, 1)
        else
            push!(smallest_first, 0)
        end

        # time until user
        time_until_user = 1
        alter_nodes = []
        for s in a
            if ego in s
                push!(time_until_users, time_until_user / length(a))
                push!(size_first_user_simplex, length(s) / avg_size_a)
                novelty = 0
                for node in s
                    if node == ego
                        continue
                    end
                    if !(node in alter_nodes)
                        novelty += 1
                    end
                end
                push!(novelty_first_user_simplex, novelty / (length(s) - 1))
                break
            else
                for node in s
                    if !(node in alter_nodes)
                        push!(alter_nodes, node)
                    end
                end
            end
            time_until_user += 1
        end

        # the first simplex is a subset of how many future sets?
        push!(first_subset, count_subsets(a, a[1])) 

        # the last simplex is a superset of how many previous sets?
        push!(last_superset, count_supersets(a, a[length(a)])) 

        # spread index
        intersections_a = []
        for j in 1:length(a)-1
            in_a = length(intersect(a[j], a[j+1]))
            push!(intersections_a, in_a)
        end
        push!(spread_index, mean(intersections_a) / avg_size_a)

        # calculate size rate of change
        push!(size_rate_of_change, (length(a[length(a)]) - length(a[1]))/length(a))

        #=
        println()
        if count == 5
            println("alternet_spread: $alternet_spread")
            println("time_until_users: $time_until_users")
            println("size_first_user_simplex: $size_first_user_simplex")
            println("novelty_first_user_simplex: $novelty_first_user_simplex")
            return
        end
        =#
        count += 1
    end

    return spread_index, smallest_first, first_subset, 
        last_superset, size_rate_of_change, lengths, alternet_spread, early_size_index, early_deg_index, 
        avg_alternet_spread_index, time_until_users, size_first_user_simplex, novelty_first_user_simplex, is_sorted
end

function make_dataframe_is_first(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, is_first = prepare_ML_set_length_data_star_is_first(len, global_sorted_dict)

    df = DataFrame(Size_a = sizes_a, Size_b = sizes_b, Total_degree_a = degrees_a, Total_degree_b = degrees_b, Num_Occurences_a = occurences_a, Num_Occurences_b = occurences_b, 
        Is_smallest_a = is_smallest_a, Is_smallest_b = is_smallest_b, Highest_degree_a = highest_max_mean_a, Highest_degree_b = highest_max_mean_b,
        Isolated_a = a_isolated_nodes, Isolated_b = b_isolated_nodes, A_subset_b = a_subset_b, B_subset_a = b_subset_a,
        Avg_sizes = avg_sizes, Avg_degrees = avg_degrees, Num_nodes = num_nodes, Is_first = is_first)

    return df
end

function make_dataframe_is_first_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, lengths, is_first = prepare_ML_set_length_data_star_is_first_all_lengths(global_sorted_dict)

    df = DataFrame(Size_a = sizes_a, Size_b = sizes_b, Total_degree_a = degrees_a, Total_degree_b = degrees_b, Num_Occurences_a = occurences_a, Num_Occurences_b = occurences_b, 
        Is_smallest_a = is_smallest_a, Is_smallest_b = is_smallest_b, Highest_degree_a = highest_max_mean_a, Highest_degree_b = highest_max_mean_b,
        Isolated_a = a_isolated_nodes, Isolated_b = b_isolated_nodes, A_subset_b = a_subset_b, B_subset_a = b_subset_a,
        Avg_sizes = avg_sizes, Avg_degrees = avg_degrees, Num_nodes = num_nodes, Lengths = lengths, Is_first = is_first)

    return df
end

function make_dataframe_a_before_b(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, a_before_b = prepare_ML_set_length_data_star_a_before_b(len, global_sorted_dict)
    #=
    for data in [sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, a_before_b]
        data = convert(Array{Float64}, data)
    end =#

    sizes_a = convert(Array{Int64}, sizes_a)
    sizes_b = convert(Array{Int64}, sizes_b)
    degrees_a = convert(Array{Int64}, degrees_a)
    degrees_b = convert(Array{Int64}, degrees_b)
    occurences_a = convert(Array{Int64}, occurences_a)
    occurences_b = convert(Array{Int64}, occurences_b)
    is_smallest_a = convert(Array{Int64}, is_smallest_a)
    is_smallest_b = convert(Array{Int64}, is_smallest_b)
    highest_max_mean_a = convert(Array{Int64}, highest_max_mean_a)
    highest_max_mean_b = convert(Array{Int64}, highest_max_mean_b)
    a_subset_b = convert(Array{Int64}, a_subset_b)
    b_subset_a = convert(Array{Int64}, b_subset_a)
    a_isolated_nodes = convert(Array{Int64}, a_isolated_nodes)
    b_isolated_nodes = convert(Array{Int64}, b_isolated_nodes)
    avg_sizes = convert(Array{Float64}, avg_sizes)
    avg_degrees = convert(Array{Float64}, avg_degrees)
    num_nodes = convert(Array{Int64}, num_nodes)
    a_before_b = convert(Array{Int64}, a_before_b)

    df = DataFrame(Size_a = sizes_a, Size_b = sizes_b, Total_degree_a = degrees_a, Total_degree_b = degrees_b, Num_Occurences_a = occurences_a, Num_Occurences_b = occurences_b, 
        Is_smallest_a = is_smallest_a, Is_smallest_b = is_smallest_b, Highest_degree_a = highest_max_mean_a, Highest_degree_b = highest_max_mean_b,
        Isolated_a = a_isolated_nodes, Isolated_b = b_isolated_nodes, A_subset_b = a_subset_b, B_subset_a = b_subset_a,
        Avg_sizes = avg_sizes, Avg_degrees = avg_degrees, Num_nodes = num_nodes, A_before_b = a_before_b)

    return df
end

function make_dataframe_a_before_b_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, lengths, intersection, a_before_b = prepare_ML_set_length_data_star_a_before_b_all_lengths(global_sorted_dict)
    #=
    for data in [sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, a_subset_b, 
        a_isolated_nodes, sizes_b, degrees_b, occurences_b, is_smallest_b, 
        highest_max_mean_b, b_subset_a, b_isolated_nodes, avg_sizes, avg_degrees, num_nodes, a_before_b]
        data = convert(Array{Float64}, data)
    end =#
    #=
    sizes_a = convert(Array{Int64}, sizes_a)
    sizes_b = convert(Array{Int64}, sizes_b)
    degrees_a = convert(Array{Int64}, degrees_a)
    degrees_b = convert(Array{Int64}, degrees_b)
    occurences_a = convert(Array{Int64}, occurences_a)
    occurences_b = convert(Array{Int64}, occurences_b)
    is_smallest_a = convert(Array{Int64}, is_smallest_a)
    is_smallest_b = convert(Array{Int64}, is_smallest_b)
    highest_max_mean_a = convert(Array{Int64}, highest_max_mean_a)
    highest_max_mean_b = convert(Array{Int64}, highest_max_mean_b)
    a_subset_b = convert(Array{Int64}, a_subset_b)
    b_subset_a = convert(Array{Int64}, b_subset_a)
    a_isolated_nodes = convert(Array{Int64}, a_isolated_nodes)
    b_isolated_nodes = convert(Array{Int64}, b_isolated_nodes)
    avg_sizes = convert(Array{Float64}, avg_sizes)
    avg_degrees = convert(Array{Float64}, avg_degrees)
    num_nodes = convert(Array{Int64}, num_nodes)
    a_before_b = convert(Array{Int64}, a_before_b)
    =#
    df = DataFrame(Size_a = sizes_a, Size_b = sizes_b, Total_degree_a = degrees_a, Total_degree_b = degrees_b, Num_Occurences_a = occurences_a, Num_Occurences_b = occurences_b, 
        Is_smallest_a = is_smallest_a, Is_smallest_b = is_smallest_b, Highest_degree_a = highest_max_mean_a, Highest_degree_b = highest_max_mean_b,
        Isolated_a = a_isolated_nodes, Isolated_b = b_isolated_nodes, A_subset_b = a_subset_b, B_subset_a = b_subset_a,
        Avg_sizes = avg_sizes, Avg_degrees = avg_degrees, Num_nodes = num_nodes, Lengths = lengths, Intersection = intersection, 
        A_before_b = a_before_b)

    return df
end

function make_dataframe_adjacent(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    sizes_a, degrees_a, occurences_a, is_smallest_a, highest_max_mean_a, 
        sizes_b, degrees_b, occurences_b, is_smallest_b, highest_max_mean_b, intersection, a_adjacent_b = 
        prepare_ML_set_length_data_star_adjacent(len, global_sorted_dict)

    df = DataFrame(Size_a = sizes_a, Size_b = sizes_b, Total_degree_a = degrees_a, Total_degree_b = degrees_b, Num_Occurences_a = occurences_a, Num_Occurences_b = occurences_b, 
        Is_smallest_a = is_smallest_a, Is_smallest_b = is_smallest_b, Highest_degree_a = highest_max_mean_a, Highest_degree_b = highest_max_mean_b,
        Intersection = intersection, A_adjacent_b = a_adjacent_b)

    return df
end

function make_dataframe_simplicies(len::Int64, global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    spread_index_a, spread_index_b, smallest_first_a, smallest_first_b, a_first_subset, b_first_subset, 
    a_last_superset, b_last_superset, size_rate_of_change_a, size_rate_of_change_b, 
    a_is_sorted = 
        prepare_ML_set_length_data_star_simplicies(len, global_sorted_dict)

    df = DataFrame(Spread_index_a = spread_index_a, Spread_index_b = spread_index_b, Smallest_first_a = smallest_first_a, 
        Smallest_first_b = smallest_first_b, A_first_subset = a_first_subset, B_first_subset = b_first_subset, 
        A_last_superset = a_last_superset, B_last_superset = b_last_superset, Size_rate_of_change_a = size_rate_of_change_a, 
        Size_rate_of_change_b = size_rate_of_change_b, A_is_sorted = a_is_sorted)

    return df
end

function make_dataframe_simplicies_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    spread_index, smallest_first, first_subset, 
        last_superset, size_rate_of_change, lengths, alternet_spread, early_size_index, early_deg_index, 
        avg_alternet_spread_index, is_sorted = 
        prepare_ML_set_length_data_star_simplicies_all_lengths(global_sorted_dict)

    df = DataFrame(Spread_index = spread_index, Smallest_first = smallest_first, First_subset = first_subset, 
        Last_superset = last_superset, Size_rate_of_change = size_rate_of_change, Length = lengths, Alternet_spread = alternet_spread, 
        Early_size_index = early_size_index, Early_deg_index = early_deg_index, 
        Avg_alternet_spread_index = avg_alternet_spread_index, Is_sorted = is_sorted)

    return df
end

function make_dataframe_simplicies_def_cont_all_lengths(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    spread_index, smallest_first, first_subset, 
        last_superset, size_rate_of_change, lengths, alternet_spread, early_size_index, early_deg_index, 
        avg_alternet_spread_index, time_until_users, size_first_user_simplex, novelty_first_user_simplex, is_sorted = 
        prepare_ML_set_length_data_def_cont_simplicies_all_lengths(global_sorted_dict)

    df = DataFrame(Spread_index = spread_index, Smallest_first = smallest_first, First_subset = first_subset, 
        Last_superset = last_superset, Size_rate_of_change = size_rate_of_change, Length = lengths, Alternet_spread = alternet_spread, 
        Avg_alternet_spread_index = avg_alternet_spread_index, Time_until_users = time_until_users, 
        Size_first_user_simplex = size_first_user_simplex, Novelty_first_user_simplex = novelty_first_user_simplex, 
        Is_sorted = is_sorted)

    return df
end

function predict_first_user_star_binary_classification_is_first(dataset::String,  len::Int64)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    # global_sorted_dict = load_sorted_dicts("coauth-DBLP", true)
    df = make_dataframe_is_first_all_lengths(global_sorted_dict)
    # df = make_dataframe(5, global_sorted_dict)
    #MLJ.@load DecisionTreeClassifier
    tree_model = DecisionTreeClassifier()
    #MLJ.@load RandomForestClassifier pkg = DecisionTree
    #tree_model = RandomForestClassifier()
    #MLJ.@load XGBoostClassifier
    #tree_model = XGBoostClassifier()   

    X = df[:, Not(:Is_first)]
    y = df[:, [:Is_first]]

    X = ((size_a=X[1], size_b=X[2], degrees_a=X[3], degrees_b=X[4], occurences_a=X[5], occurences_b=X[6], 
        is_smallest_a=X[7], is_smallest_b=X[8], highest_max_mean_a=X[9], highest_max_mean_b=X[10], 
        iso_a=X[11], iso_b=X[12], a_sub_b=X[13], b_sub_a=X[14], avg_size=X[15], avg_deg=X[16], num_nodes=X[17]))
    y = convert(CategoricalArray, convert(Array{Int64}, y[1]))

    tree = machine(tree_model, X, y)

    train, test = partition(eachindex(y), 0.9, shuffle=true)
    MLJ.fit!(tree, rows=train)

    y_pred = MLJ.predict(tree, rows=test)
    ȳ = predict_mode(tree, rows=test)
    accuracy = sum(ȳ .== y[test]) / length(y[test])
    println("accuracy: $accuracy")
    #=
    1:18
    Train, test = partition(eachindex(y), 0.9, shuffle=true)

    tree = machine(tree_model, X, y)
    MLJ.fit!(tree, rows=Train)
    y_pred = MLJ.predict(tree, rows=test)
    ȳ = predict_mode(tree, rows=test)
    Accuracy = sum(ȳ .== y[test]) / length(y[test])
    println("accuracy: $Accuracy")
    mce = cross_entropy(y_pred, y[test]) |> mean
    println("mce: $mce")
    
    =#

    mce = cross_entropy(y_pred, y[test]) |> mean
    println("mce: $mce")

    # bar chart with different accuracies
end

function predict_first_user_star_binary_classification_a_before_b(dataset::String, len::Int64)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    df = make_dataframe_a_before_b(len, global_sorted_dict)

    MLJ.@load DecisionTreeClassifier
    tree_model = DecisionTreeClassifier()  
    #MLJ.@load XGBoostClassifier
    #tree_model = XGBoostClassifier()  

    X = df[:, Not(:A_before_b)]
    y = df[:, [:A_before_b]]

    X = ((size_a=X[1], size_b=X[2], degrees_a=X[3], degrees_b=X[4], occurences_a=X[5], occurences_b=X[6], 
        is_smallest_a=X[7], is_smallest_b=X[8], highest_max_mean_a=X[9], highest_max_mean_b=X[10], 
        iso_a=X[11], iso_b=X[12], a_sub_b=X[13], b_sub_a=X[14], avg_size=X[15], avg_deg=X[16], num_nodes=X[17]))
    y = convert(CategoricalArray, convert(Array{Int64}, y[1]))

    tree = machine(tree_model, X, y)

    train, test = partition(eachindex(y), 0.9, shuffle=true)

    MLJ.fit!(tree, rows=train)

    y_pred = MLJ.predict(tree, rows=test)
    ȳ = predict_mode(tree, rows=test)
    accuracy = sum(ȳ .== y[test]) / length(y[test])
    println("accuracy: $accuracy")

    mce = cross_entropy(y_pred, y[test]) |> mean
    println("mce: $mce")

    #=

    Train, test = partition(eachindex(y), 0.9, shuffle=true)

    tree = machine(tree_model, X, y)
    MLJ.fit!(tree, rows=Train)
    y_pred = MLJ.predict(tree, rows=test)
    ȳ = predict_mode(tree, rows=test)
    Accuracy = sum(ȳ .== y[test]) / length(y[test])
    println("accuracy: $Accuracy")
    mce = cross_entropy(y_pred, y[test]) |> mean
    println("mce: $mce")
    
    --------------------------

    # C=1.4, 0.11
    # names(df)
    # X = convert(Array, df[:, [:Size_a, :Size_b]])
    # accuracy = sum(ScikitLearn.predict(model, X) .== y) / length(y)

    X = convert(Array, df2[:, Not(:A_before_b)])
    y = df2[:, [:A_before_b]][1] # y = convert(Array, df[:, [:A_before_b]])
    model = LogisticRegression(fit_intercept=true, max_iter=200)
    ScikitLearn.fit!(model, X, y)
    
    using ScikitLearn.CrossValidation: cross_val_score
    accuracies = cross_val_score(model, X, y; cv=10)
    mean_accuracy = mean(accuracies)
    std_accuracies = std(accuracies)
    println("accuracy: $mean_accuracy +- $std_accuracies")

    using ScikitLearn.GridSearch: GridSearchCV
    gridsearch = GridSearchCV(LogisticRegression(fit_intercept=true, max_iter=200), Dict(:C => 0.1:0.1:2.0))
    ScikitLearn.fit!(gridsearch, X, y)
    println("Best hyper-parameters: $(gridsearch.best_params_)")

    using ScikitLearn
    using ScikitLearn.Pipelines: Pipeline, make_pipeline
    @sk_import decomposition: PCA

    estimators = [("reduce_dim", PCA()), ("logistic_regression", LogisticRegression())]
    clf = Pipeline(estimators)
    fit!(clf, X, y)

    using PyPlot

    plot([cv_res.parameters[:C] for cv_res in gridsearch.grid_scores_],
        mean(cv_res.cv_validation_scores) for cv_res in gridsearch.grid_scores_])

    =#

    # bar chart with different accuracies
end

function predict_first_user_star_binary_classification_adjacent(dataset::String, len::Int64)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    df = make_dataframe_adjacent(len, global_sorted_dict)

    #MLJ.@load DecisionTreeClassifier
    tree_model = DecisionTreeClassifier()  
    #MLJ.@load XGBoostClassifier
    #tree_model = XGBoostClassifier()   

    #X = df[:, [:Size, :Max_mean_degree]]
    #X = df[:, Not(:Is_first)]
    #y = df[:, [:Is_first]]
    #X = df[:, Not(:A_before_b)]
    #y = df[:, [:A_before_b]]
    X = df[:, Not(:A_adjacent_b)]
    y = df[:, [:A_adjacent_b]]

    #X = ((size=X[1], deg=X[2]))
    X = ((size_a=X[1], size_b=X[2], degrees_a=X[3], degrees_b=X[4], occurences_a=X[5], 
        occurences_b=X[6], is_smallest_a=X[7], is_smallest_b=X[8], highest_max_mean_a=X[9], 
        highest_max_mean_b=X[10], adj=X[11]))
    #X = ((size1=X[1], size2=X[2], size3=X[3], size4=X[4], size5=X[5], degree1=X[6], degree2=X[7], degree3=X[8], degree4=X[9], degree5=X[10]))
    y = convert(CategoricalArray, convert(Array{Int64}, y[1]))

    tree = machine(tree_model, X, y)

    train, test = partition(eachindex(y), 0.9, shuffle=true)

    MLJ.fit!(tree, rows=train)

    y_pred = MLJ.predict(tree, rows=test)
    ȳ = predict_mode(tree, rows=test)
    accuracy = sum(ȳ .== y[test]) / length(y[test])
    println("accuracy: $accuracy")

    mce = cross_entropy(y_pred, y[test]) |> mean
    println("mce: $mce")

    #=
    # X = ((size_a=X[1], size_b=X[2], degrees_a=X[3], degrees_b=X[4], occurences_a=X[5], occurences_b=X[6], is_smallest_a=X[7], is_smallest_b=X[8], highest_max_mean_a=X[9], highest_max_mean_b=X[10]))

    length([e for e in df.A_adjacent_b if e == 1]) / length(df.A_adjacent_b)

    tree = machine(tree_model, X, y)
    MLJ.fit!(tree, rows=Train)
    y_pred = MLJ.predict(tree, rows=test)
    ȳ = predict_mode(tree, rows=test)
    Accuracy = sum(ȳ .== y[test]) / length(y[test])
    println("accuracy: $Accuracy")
    mce = cross_entropy(y_pred, y[test]) |> mean
    println("mce: $mce")
    
    =#

    # bar chart with different accuracies
end

function predict_is_sorted(dataset::String, len::Int64)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    df = make_dataframe_simplicies(len, global_sorted_dict)

    # X = convert(Array, df[:, [:Size_a, :Size_b]])
    X = convert(Array, df[:, Not(:A_is_sorted)])
    y = df[:, [:A_is_sorted]][1]
    model = LogisticRegression(fit_intercept=true, max_iter=200)
    ScikitLearn.fit!(model, X, y)
    
    accuracies = cross_val_score(model, X, y; cv=10)
    mean_accuracy = mean(accuracies)
    std_accuracies = std(accuracies)
    println("accuracy: $mean_accuracy ± $std_accuracies")
    #=
    gridsearch = GridSearchCV(LogisticRegression(fit_intercept=true, max_iter=200), Dict(:C => 0.1:0.1:2.0))
    ScikitLearn.fit!(gridsearch, X, y)
    println("Best hyper-parameters: $(gridsearch.best_params_)")

    @sk_import decomposition: PCA

    estimators = [("reduce_dim", PCA()), ("logistic_regression", LogisticRegression())]
    clf = Pipeline(estimators)
    fit!(clf, X, y)
    =#
end

function train(#=X, y, model, train_test_split=#)

end

function spread_similar_simplicies_star(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    lengths = 10:20
    #global_sorted_dict = load_sorted_dicts(dataset, true)
    spread_dict = DataStructures.SortedDict()

    for len in lengths
        print(stdout, "Length: $len \r")
        flush(stdout)
        spread_dict[len] = []
        actual_spread = []
        shuffled_spread = []

        for (ego, info) in global_sorted_dict
            if info[7] == len
                egonet = info[3]
                simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
                #shuffle!(simplicies)
                #println("Simplicies: $simplicies")
                for simplex in simplicies
                    #println("Simplex: $simplex")
                    push!(simplex, -1)
                    similar_simplicies = get_similar_simplicies(simplicies, simplex)
                    #println("Similar Simplicies: $similar_simplicies")
                    #println()
                    for s in similar_simplicies
                        #println("s: $s")
                        push!(s, -2)
                        shuffled_distances_s = []
                        actual_distance_s = get_distances(simplicies, simplex, s)
                        #println("Actual Distanct: $actual_distance_s")
                        #println()
                        for i in 1:100
                            #println("i: $i")
                            shuffled_simplicies = shuffle(simplicies)
                            shuffled_distance = get_distances(shuffled_simplicies, simplex, s)
                            #println("Shuffled Distance: $shuffled_distance")
                            #println("Shuffled Distances s: $shuffled_distances_s")
                            push!(shuffled_distances_s, shuffled_distance)
                        end
                    
                        avg_shuffled_distance_s = mean(shuffled_distances_s) # could use median?
                        pop!(s)

                        push!(actual_spread, actual_distance_s)
                        push!(shuffled_spread, avg_shuffled_distance_s)
                        #println()
                    end

                    pop!(simplex)
                end
            end
        end

        push!(spread_dict[len], mean(actual_spread))
        push!(spread_dict[len], mean(shuffled_spread))
    end

    X = lengths
    Y_actual = []
    Y_shuffle = []

    for (k, v) in spread_dict
        push!(Y_actual, v[1])
        push!(Y_shuffle, v[2])
    end

    Plots.plot(X, [Y_actual, Y_shuffle], xlabel = "length of star egonet", ylabel = "average spread", label = ["Actual spread" "Shuffled spread"], left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Average Spread for Similar Simplicies - Star", linewidth=0.5)
    Plots.savefig("coauth-DBLP_star_spread_similar_all.pdf")

    return

end

function spread_duplicate_simplicies_star(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    lengths = 10:20
    #global_sorted_dict = load_sorted_dicts(dataset, true)
    spread_dict = DataStructures.SortedDict()

    for len in lengths
        print(stdout, "Length: $len \r")
        flush(stdout)
        spread_dict[len] = []
        actual_spread = []
        shuffled_spread = []

        for (ego, info) in global_sorted_dict
            if info[7] == len
                egonet = info[3]
                simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
                
                for simplex in simplicies
                    sort!(simplex)
                end

                #println("Simplicies: $simplicies")

                for simplex in simplicies
                    simplex_copy = deepcopy(simplex)
                    push!(simplex, -1)
                    #println("Simplex copy: $simplex_copy")

                    if simplex_copy in simplicies
                        #println("There's a duplicate! Simplex: $simplex")

                        shuffled_distances_s = []
                        actual_distance_s = get_distances(simplicies, simplex, simplex_copy)
                        #println("Actual Distanct: $actual_distance_s")
                        for i in 1:100
                            #println("i: $i")
                            shuffled_simplicies = shuffle(simplicies)
                            shuffled_distance = get_distances(shuffled_simplicies, simplex, simplex_copy)

                            push!(shuffled_distances_s, shuffled_distance)
                            #println("Shuffled Distance: $shuffled_distance")
                            #println("Shuffled Distances s: $shuffled_distances_s")
                        end
                    
                        avg_shuffled_distance_s = mean(shuffled_distances_s) # could use median?
                        #pop!(s)

                        push!(actual_spread, actual_distance_s)
                        push!(shuffled_spread, avg_shuffled_distance_s)
                        #println()

                        pop!(simplex)
                    end
                end
            end
        end

        push!(spread_dict[len], mean(actual_spread))
        push!(spread_dict[len], mean(shuffled_spread))
    end

    X = lengths
    Y_actual = []
    Y_shuffle = []

    for (k, v) in spread_dict
        push!(Y_actual, v[1])
        push!(Y_shuffle, v[2])
    end

    Plots.plot(X, [Y_actual, Y_shuffle], xlabel = "length of star egonet", ylabel = "average spread", label = ["Actual spread" "Shuffled spread"], left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Actual vs Shuffled Average Spread for Duplicate Simplicies - Star", linewidth=0.5)
    Plots.savefig("coauth-DBLP_star_spread_duplicate.pdf")

    return spread_dict

end

function spread_similar_simplicies_def_cont(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    lengths = 10:50
    #global_sorted_dict = load_sorted_dicts(dataset, true)
    spread_dict = DataStructures.SortedDict()

    for len in lengths
        print(stdout, "Length: $len \r")
        flush(stdout)
        spread_dict[len] = []
        actual_spread = []
        shuffled_spread = []

        for (ego, info) in global_sorted_dict
            if info[5] == len
                egonet = info[1]
                #simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
                simplicies = network_to_list(egonet)
                #shuffle!(simplicies)
                #println("Simplicies: $simplicies")
                for simplex in simplicies
                    #println("Simplex: $simplex")
                    push!(simplex, -1)
                    similar_simplicies = get_similar_simplicies(simplicies, simplex)
                    #println("Similar Simplicies: $similar_simplicies")
                    #println()
                    for s in similar_simplicies
                        #println("s: $s")
                        push!(s, -2)
                        shuffled_distances_s = []
                        actual_distance_s = get_distances(simplicies, simplex, s)
                        #println("Actual Distanct: $actual_distance_s")
                        #println()
                        for i in 1:100
                            #println("i: $i")
                            shuffled_simplicies = shuffle(simplicies)
                            shuffled_distance = get_distances(shuffled_simplicies, simplex, s)
                            #println("Shuffled Distance: $shuffled_distance")
                            #println("Shuffled Distances s: $shuffled_distances_s")
                            push!(shuffled_distances_s, shuffled_distance)
                        end
                    
                        avg_shuffled_distance_s = mean(shuffled_distances_s) # could use median?
                        pop!(s)

                        push!(actual_spread, actual_distance_s)
                        push!(shuffled_spread, avg_shuffled_distance_s)
                        #println()
                    end

                    pop!(simplex)
                end
            end
        end

        push!(spread_dict[len], mean(actual_spread))
        push!(spread_dict[len], mean(shuffled_spread))
    end

    X = lengths
    Y_actual = []
    Y_shuffle = []

    for (k, v) in spread_dict
        push!(Y_actual, v[1])
        push!(Y_shuffle, v[2])
    end

    Plots.plot(X, [Y_actual, Y_shuffle], xlabel = "length of default egonet", ylabel = "average spread", label = ["Actual spread" "Shuffled spread"], left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Actual vs Shuffled Average Spread for Similar Simplicies - Default", linewidth=0.5)
    Plots.savefig("coauth-DBLP_def_spread_similar.pdf")


    return spread_dict

end

function spread_duplicate_simplicies_def_cont(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    lengths = 10:50
    #global_sorted_dict = load_sorted_dicts(dataset, true)
    spread_dict = DataStructures.SortedDict()

    for len in lengths
        print(stdout, "Length: $len \r")
        flush(stdout)
        spread_dict[len] = []
        actual_spread = []
        shuffled_spread = []

        for (ego, info) in global_sorted_dict
            if info[5] == len
                egonet = info[1]
                #simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
                simplicies = network_to_list(egonet)
                for simplex in simplicies
                    sort!(simplex)
                end

                #println("Simplicies: $simplicies")

                for simplex in simplicies
                    simplex_copy = deepcopy(simplex)
                    push!(simplex, -1)
                    #println("Simplex copy: $simplex_copy")

                    if simplex_copy in simplicies
                        #println("There's a duplicate! Simplex: $simplex")

                        shuffled_distances_s = []
                        actual_distance_s = get_distances(simplicies, simplex, simplex_copy)
                        #println("Actual Distanct: $actual_distance_s")
                        for i in 1:100
                            #println("i: $i")
                            shuffled_simplicies = shuffle(simplicies)
                            shuffled_distance = get_distances(shuffled_simplicies, simplex, simplex_copy)

                            push!(shuffled_distances_s, shuffled_distance)
                            #println("Shuffled Distance: $shuffled_distance")
                            #println("Shuffled Distances s: $shuffled_distances_s")
                        end
                    
                        avg_shuffled_distance_s = mean(shuffled_distances_s) # could use median?
                        #pop!(s)

                        push!(actual_spread, actual_distance_s)
                        push!(shuffled_spread, avg_shuffled_distance_s)
                        #println()

                        pop!(simplex)
                    end
                end
            end
        end

        push!(spread_dict[len], mean(actual_spread))
        push!(spread_dict[len], mean(shuffled_spread))
    end

    X = lengths
    Y_actual = []
    Y_shuffle = []

    for (k, v) in spread_dict
        push!(Y_actual, v[1])
        push!(Y_shuffle, v[2])
    end

    Plots.plot(X, [Y_actual, Y_shuffle], xlabel = "length of default egonet", ylabel = "average spread", label = ["Actual spread" "Shuffled spread"], left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Actual vs Shuffled Average Spread for Duplicate Simplicies - Default", linewidth=0.5)
    Plots.savefig("coauth-DBLP_def_spread_duplicate.pdf")


    return spread_dict

end

function plot_absolute_simplex_size(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    global_size_dict = DataStructures.SortedDict()
    X = []
    Y = []

    for (ego, info) in global_sorted_dict
        print(stdout, "Ego: $ego \r")
        flush(stdout)

        egonet = info[1] #2
        star_count_dict = DataStructures.SortedDict()

        data_tie_strength, data_n, data_size, data_degrees, data_ratio, count, final_index, avg_novelty_rate, avg_size, 
            frac_simplices_before_ego = set_data_absolute_time_no_user(egonet, ego, 0, 0, star_count_dict, 0)

        for (k, v) in data_size
            try
                push!(global_size_dict[k], v)
            catch e
                global_size_dict[k] = [v]
            end
        end
    end

    for (k, v) in global_size_dict
        push!(X, k)
        push!(Y, mean(v))
    end

    Plots.plot(X, Y, xlabel = "absolute time", ylabel = "average incoming simplex size", left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "Average incoming simplex size over absolute time", linewidth=0.5)
    Plots.savefig("$(dataset)_absolute_size.pdf")
end

function plot_simplex_with_ego_stats(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    star_lengths = [info[7] for (ego, info) in global_sorted_dict]
    sort!(star_lengths)
    unique!(star_lengths)
    X = 5:100
    #X = star_lengths
    Y_simp_ego = []
    Y_deg = []
    Y_simp_ego_cont = []
    Y_deg_cont = []

    Y_size = []
    Y_size_cont = []
    Y_simp_ego_size = []
    Y_simp_ego_size_cont = []

    Y_novelty = []
    Y_novelty_cont = []
    Y_simp_ego_novelty = []
    Y_simp_ego_novelty_cont = []

    Y_time = []
    Y_time_cont = []

    for length in X
        print(stdout, "Length: $length \r")
        flush(stdout)

        simplex_with_ego_to_average = []
        degree_to_average = []
        simplex_with_ego_to_average_cont = []
        degree_to_average_cont = []

        size_to_average = []
        size_to_average_cont = []
        simplex_with_ego_size_to_average = []
        simplex_with_ego_size_to_average_cont = []

        novelty_to_average = []
        novelty_to_average_cont = []
        simplex_with_ego_novelty_to_average = []
        simplex_with_ego_novelty_to_average_cont = []

        time_to_average = []
        time_to_average_cont = []

        for (ego, info) in global_sorted_dict
            #def
            if info[5] == length #6
                egonet = info[1] #2
                star_count_dict = DataStructures.SortedDict()
                
                avg_degree, avg_deg_simplex_with_ego, avg_size, simplex_with_ego_size, avg_novelty, simplex_with_ego_novelty, index = 
                    set_data_ordinal_time_until_user(egonet, ego, 0, 0, star_count_dict, 0)
                #=
                push!(simplex_with_ego_to_average, avg_deg_simplex_with_ego)
                push!(degree_to_average, avg_degree)
                =#
                #=
                push!(simplex_with_ego_size_to_average, simplex_with_ego_size)
                push!(size_to_average, avg_size)
                =#
                
                push!(simplex_with_ego_novelty_to_average, simplex_with_ego_novelty)
                push!(novelty_to_average, avg_novelty)
                
                push!(time_to_average, index)

            end
            #cont
            if info[6] == length #6
                egonet = info[2] #2
                star_count_dict = DataStructures.SortedDict()
                
                avg_degree, avg_deg_simplex_with_ego, avg_size, simplex_with_ego_size, avg_novelty, simplex_with_ego_novelty, index = 
                    set_data_ordinal_time_until_user(egonet, ego, 0, 0, star_count_dict, 0)
                #=
                push!(simplex_with_ego_to_average_cont, avg_deg_simplex_with_ego)
                push!(degree_to_average_cont, avg_degree)
                =#
                #=
                push!(simplex_with_ego_size_to_average_cont, simplex_with_ego_size)
                push!(size_to_average_cont, avg_size)
                =#
                
                push!(simplex_with_ego_novelty_to_average_cont, simplex_with_ego_novelty)
                push!(novelty_to_average_cont, avg_novelty)
                
                push!(time_to_average_cont, index)
            end
        end
        #=
        push!(Y_simp_ego, mean(simplex_with_ego_to_average))
        push!(Y_deg, mean(degree_to_average))

        push!(Y_simp_ego_cont, mean(simplex_with_ego_to_average_cont))
        push!(Y_deg_cont, mean(degree_to_average_cont))
        =#
        #=
        push!(Y_simp_ego_size, mean(simplex_with_ego_size_to_average))
        push!(Y_size, mean(size_to_average))

        push!(Y_simp_ego_size_cont, mean(simplex_with_ego_size_to_average_cont))
        push!(Y_size_cont, mean(size_to_average_cont))
        =#
        
        push!(Y_simp_ego_novelty, mean(simplex_with_ego_novelty_to_average))
        push!(Y_novelty, mean(novelty_to_average))

        push!(Y_simp_ego_novelty_cont, mean(simplex_with_ego_novelty_to_average_cont))
        push!(Y_novelty_cont, mean(novelty_to_average_cont))
        
        push!(Y_time, mean(time_to_average))
        push!(Y_time_cont, mean(time_to_average_cont))
    end
    #=
    SAVE

    Plots.plot(X, [Y_simp_ego, Y_deg], xlabel = "length of egonet x", ylabel = "degree", label = ["Degree of first simplex with ego" "Avg degree until ego"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Default degrees", linewidth=0.5)
    Plots.savefig("$(dataset)_def_degrees_until_user.pdf")

    Plots.plot(X, [Y_simp_ego_cont, Y_deg_cont], xlabel = "length of egonet x", ylabel = "degree", label = ["Degree of first simplex with ego" "Avg degree until ego"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Contracted Degrees", linewidth=0.5)
    Plots.savefig("$(dataset)_cont_degrees_until_user.pdf")
    =#
    #=
    FileIO.save("../../../data/clc348/def_$(dataset)_until_user_size_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_until_user_size_Y.jld2","Y_size",Y_size)
    
    FileIO.save("../../../data/clc348/cont_$(dataset)_until_user_size_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_until_user_size_Y.jld2","Y_size_cont",Y_size_cont)

    FileIO.save("../../../data/clc348/def_$(dataset)_with_user_size_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_with_user_size_Y.jld2","Y_simp_ego_size",Y_simp_ego_size)
    
    FileIO.save("../../../data/clc348/cont_$(dataset)_with_user_size_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_with_user_size_Y.jld2","Y_simp_ego_size_cont",Y_simp_ego_size_cont)

    Plots.plot(X, [Y_simp_ego_size, Y_size], xlabel = "length of egonet x", ylabel = "incoming simplex size", label = ["Size of first simplex with ego" "Avg simplex size until ego"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Default simplex size", linewidth=0.5)
    Plots.savefig("$(dataset)_def_size_until_user.pdf")

    Plots.plot(X, [Y_simp_ego_size_cont, Y_size_cont], xlabel = "length of egonet x", ylabel = "incoming simplex size", label = ["Size of first simplex with ego" "Avg simplex size until ego"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Contracted simplex size", linewidth=0.5)
    Plots.savefig("$(dataset)_cont_size_until_user.pdf")
    =#
    
    FileIO.save("../../../data/clc348/def_$(dataset)_until_user_novelty_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_until_user_novelty_Y.jld2","Y_novelty",Y_novelty)
    
    FileIO.save("../../../data/clc348/cont_$(dataset)_until_user_novelty_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_until_user_novelty_Y.jld2","Y_novelty_cont",Y_novelty_cont)

    FileIO.save("../../../data/clc348/def_$(dataset)_with_user_novelty_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_with_user_novelty_Y.jld2","Y_simp_ego_novelty",Y_simp_ego_novelty)
    
    FileIO.save("../../../data/clc348/cont_$(dataset)_with_user_novelty_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_with_user_novelty_Y.jld2","Y_simp_ego_novelty_cont",Y_simp_ego_novelty_cont)
  
    Plots.plot(X, [Y_simp_ego_novelty, Y_novelty], xlabel = "length of egonet x", ylabel = "novelty", label = ["Novelty of user simplex (minus user)" "Avg novelty until user simplex"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Default novelty", linewidth=0.5)
    Plots.savefig("$(dataset)_def_novelty_until_user.pdf")

    Plots.plot(X, [Y_simp_ego_novelty_cont, Y_novelty_cont], xlabel = "length of egonet x", ylabel = "novelty", label = ["Novelty of user simplex (minus user)" "Avg novelty until user simplex"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Contracted novelty", linewidth=0.5)
    Plots.savefig("$(dataset)_cont_novelty_until_user.pdf")
    
    #=
    FileIO.save("../../../data/clc348/$(dataset)_time_until_user_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_time_until_user_Y.jld2","Y_time",Y_time)
    FileIO.save("../../../data/clc348/cont_$(dataset)_time_until_user_Y.jld2","Y_time_cont",Y_time_cont)

    Plots.plot(X, [Y_time, Y_time_cont], xlabel = "length of egonet x", ylabel = "time until user", label = ["Default" "Contracted"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "Time until user", linewidth=0.5)
    Plots.savefig("$(dataset)_time_until_user.pdf")
    =#
    
end

function plot_values_for_lengths_def_cont(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    X = 5:100
    Y_tie = []
    Y_n = []
    Y_size = []
    Y_deg = []
    Y_ratio = []
    Y_avg_novelty = []
    Y_avg_size = []
    Y_frac = []
    Y_first_size = []

    Y_tie_cont = []
    Y_n_cont = []
    Y_size_cont = []
    Y_deg_cont = []
    Y_ratio_cont = []
    Y_avg_novelty_cont = []
    Y_avg_size_cont = []
    Y_frac_cont = []
    Y_first_size_cont = []
    Y_frac_cont = []
    
    for length in X
        print(stdout, "Length: $length \r")
        flush(stdout)

        novelty_to_average = []
        novelty_to_average_cont = []

        size_to_average = []
        size_to_average_cont = []

        n_to_average = []
        n_to_average_cont = []
        #def
        for (ego, info) in global_sorted_dict
            if info[5] == length
                egonet = info[1]
                star_count_dict = DataStructures.SortedDict()
                
                data_tie_strength, data_n, data_size, data_degrees, data_ratio, count, final_index, avg_novelty_rate, avg_size, 
                    frac_simplices_before_ego, count_until_ego = set_data_ordinal_time_no_user(egonet, ego, 0, 0, star_count_dict, 0)
                #if !(isnan(frac_simplices_before_ego))
                #    push!(frac_to_average, frac_simplices_before_ego)
                #end
                #=
                push!(novelty_to_average, avg_novelty_rate)
                push!(size_to_average, avg_size)
                =#
                push!(n_to_average, data_n[length])
            end
            #cont
            if info[6] == length
                egonet = info[2] 
                star_count_dict = DataStructures.SortedDict()
                
                data_tie_strength, data_n, data_size, data_degrees, data_ratio, count, final_index, avg_novelty_rate, avg_size, 
                    frac_simplices_before_ego_cont, count_until_ego = set_data_ordinal_time_no_user(egonet, ego, 0, 0, star_count_dict, 0)
                #if !(isnan(frac_simplices_before_ego_cont))
                #    push!(frac_to_average_cont, frac_simplices_before_ego_cont)
                #end
                #=
                push!(novelty_to_average_cont, avg_novelty_rate)
                push!(size_to_average_cont, avg_size)
                =#
                push!(n_to_average_cont, data_n[length])
            end
        end
        #=
        push!(Y_avg_size, mean(size_to_average))
        push!(Y_avg_size_cont, mean(size_to_average_cont))

        push!(Y_avg_novelty, mean(novelty_to_average))
        push!(Y_avg_novelty_cont, mean(novelty_to_average_cont))
        =#
        push!(Y_n, mean(n_to_average))
        push!(Y_n_cont, mean(n_to_average_cont))
    end
    #=
    FileIO.save("../../../data/clc348/def_$(dataset)_avg_novelty_rate_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_avg_novelty_rate_Y.jld2","Y_avg_novelty",Y_avg_novelty)

    FileIO.save("../../../data/clc348/cont_$(dataset)_avg_novelty_rate_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_avg_novelty_rate_Y.jld2","Y_avg_novelty_cont",Y_avg_novelty_cont)

    Plots.plot(X, [Y_avg_novelty, Y_avg_novelty_cont], xlabel = "length of egonet x", ylabel = "average novelty rate", label = ["Default" "Contracted"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "average novelty rate of default and contracted egonets", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_novelty_rate.pdf")

    FileIO.save("../../../data/clc348/def_$(dataset)_avg_simplex_size_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_avg_simplex_size_Y.jld2","Y_avg_size",Y_avg_size)

    FileIO.save("../../../data/clc348/cont_$(dataset)_avg_simplex_size_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_avg_simplex_size_Y.jld2","Y_avg_size_cont",Y_avg_size_cont)

    Plots.plot(X, [Y_avg_size, Y_avg_size_cont], xlabel = "length of egonet x", ylabel = "average simplex size", label = ["Default" "Contracted"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "average incoming simplex size of default and contracted egonets", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_simplex_size.pdf")
    =#

    #=
    FileIO.save("../../../data/clc348/def_$(dataset)_avg_final_n_X.jld2","X",X)
    FileIO.save("../../../data/clc348/def_$(dataset)_avg_final_n_Y.jld2","Y_n",Y_n)

    FileIO.save("../../../data/clc348/cont_$(dataset)_avg_final_n_X.jld2","X",X)
    FileIO.save("../../../data/clc348/cont_$(dataset)_avg_final_n_Y.jld2","Y_n_cont",Y_n_cont)

    Plots.plot(X, [Y_n, Y_n_cont], xlabel = "length of egonet x", ylabel = "average final number of nodes", label = ["Default" "Contracted"], xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = true, title = "average final number of nodes of default and contracted egonets", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_final_n.pdf")
    =#
end 

function plot_values_for_lengths_star(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    star_lengths = [info[7] for (ego, info) in global_sorted_dict]
    sort!(star_lengths)
    unique!(star_lengths)
    X = 5:100
    #X = star_lengths
    Y_tie = []
    Y_n = []
    Y_size = []
    Y_deg = []
    Y_ratio = []
    Y_novelty = []
    Y_avg_size = []
    Y_frac = []
    Y_first_size = []
    Y_frac_cont = []

    
    for length in X
        #print(stdout, "Length: $length \r")
        #flush(stdout)
        println("Length: $length")

        ts_to_average = []
        n_to_average = []
        size_to_average = []
        first_size_to_average = []
        deg_to_average = []
        ratio_to_average = []
        novelty_to_average = []
        avg_size_to_average = []
        frac_to_average = []
        for (ego, info) in global_sorted_dict
            if info[7] == length
                egonet = info[3]
                #println("ego: $ego")
                star_count_dict = DataStructures.SortedDict()
                
                data_tie_strength, data_n, data_size, data_degrees, data_ratio, count, final_index, avg_novelty_rate, avg_size, 
                    frac_simplices_before_ego = set_data_ordinal_time_no_user(egonet, ego, 0, 0, star_count_dict, 0)
                #println(frac_simplices_before_ego)
                return
                #push!(avg_size_to_average, avg_size)
                #push!(novelty_to_average, avg_novelty_rate)
                #push!(first_size_to_average, data_size[1])
                #push!(frac_to_average, frac_simplices_before_ego)
                #=
                push!(ts_to_average, data_tie_strength[length])
                push!(n_to_average, data_n[length])
                push!(size_to_average, data_size[length])
                push!(deg_to_average, data_degrees[length])
                push!(ratio_to_average, data_ratio[length])
                =#
                #println("Here")
            end
        end
        #push!(Y_avg_size, mean(avg_size_to_average))
        #push!(Y_novelty, mean(novelty_to_average))
        #push!(Y_frac, mean(frac_to_average))
        #push!(Y_first_size, mean(first_size_to_average))
        #=
        push!(Y_tie, mean(ts_to_average))
        push!(Y_n, mean(n_to_average))
        push!(Y_size, mean(size_to_average))
        push!(Y_deg, mean(deg_to_average))
        push!(Y_ratio, mean(ratio_to_average))
        =#
    end
    
    #Plots.plot(X, Y_avg_size, xlabel = "length of star egonet x", ylabel = "average incoming simplex size of egonets of length x", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average incoming simplex size of egos (no user)", linewidth=0.5)
    #Plots.savefig("$(dataset)_avg_lengths_avg_size_no_user.pdf")

    #Plots.plot(X, Y_novelty, xlabel = "length of star egonet x", ylabel = "average novelty rate of egonets of length x", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average novelty rates of egos (no user)", linewidth=0.5)
    #Plots.savefig("$(dataset)_avg_lengths_novelty_rate_no_user.pdf")
    
    #Plots.plot(X, Y_first_size, xlabel = "length of star egonet x", ylabel = "average first value of simplex size", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average first values of simplex size (no user)", linewidth=0.5)
    #Plots.savefig("$(dataset)_avg_lengths_first_size_no_user.pdf")

    #=
    Plots.plot(X, Y_tie, xlabel = "length of star egonet x", ylabel = "average final value of tie strength", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average final values of tie strength (no user)", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_lengths_tie_strength_no_user.pdf")

    Plots.plot(X, Y_n, xlabel = "length of star egonet x", ylabel = "average final value of n", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average final values of number of nodes (no user)", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_lengths_n_no_user.pdf")

    Plots.plot(X, Y_size, xlabel = "length of star egonet x", ylabel = "average final value of simplex size", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average final values of simplex size (no user)", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_lengths_size_no_user.pdf")

    Plots.plot(X, Y_deg, xlabel = "length of star egonet x", ylabel = "average final value of degree", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average final values of degree (no user)", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_lengths_degree_no_user.pdf")

    Plots.plot(X, Y_ratio, xlabel = "length of star egonet x", ylabel = "average final value of ratio", xlim=(5, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "average final values of ratio (no user)", linewidth=0.5)
    Plots.savefig("$(dataset)_avg_lengths_ratio_no_user.pdf")
    =#
end

function plot_email_values(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    lengths = [info[7] for (ego, info) in global_sorted_dict]
    #sort!(star_lengths)
    unique!(lengths)
    sort!(lengths)
    println("Lengths: $lengths")
    def_cont_len_dict = DataStructures.SortedDict()
    star_len_dict = DataStructures.SortedDict()
    for l in lengths
        if 50 <= l < 100
            try
                push!(star_len_dict[50], l)
            catch e
                star_len_dict[50] = [l]
            end
        elseif 100 <= l < 150
            try
                push!(star_len_dict[100], l)
            catch e
                star_len_dict[100] = [l]
            end
        elseif 150 <= l < 200
            try
                push!(star_len_dict[150], l)
            catch e
                star_len_dict[150] = [l]
            end
        elseif 200 <= l < 300
            try
                push!(star_len_dict[200], l)
            catch e
                star_len_dict[200] = [l]
            end
        elseif 300 <= l < 500
            try
                push!(star_len_dict[300], l)
            catch e
                star_len_dict[300] = [l]
            end
        elseif 500 <= l < 1000
            try
                push!(def_cont_len_dict[500], l)
            catch e
                def_cont_len_dict[500] = [l]
            end
        elseif 1000 <= l < 1500
            try
                push!(def_cont_len_dict[1000], l)
            catch e
                def_cont_len_dict[1000] = [l]
            end
        elseif 300 <= l < 500
            try
                push!(def_cont_len_dict[300], l)
            catch e
                def_cont_len_dict[300] = [l]
            end
        elseif 1500 <= l < 2000
            try
                push!(def_cont_len_dict[1500], l)
            catch e
                def_cont_len_dict[1500] = [l]
            end
        elseif 2000 <= l < 3000
            try
                push!(def_cont_len_dict[2000], l)
            catch e
                def_cont_len_dict[2000] = [l]
            end
        elseif 3000 <= l < 4000
            try
                push!(def_cont_len_dict[3000], l)
            catch e
                def_cont_len_dict[3000] = [l]
            end
        elseif l >= 4000
            try
                push!(def_cont_len_dict[4000], l)
            catch e
                def_cont_len_dict[4000] = [l]
            end
        end
    end
    #=
    for (k, v) in def_cont_len_dict
        #println("Time: $k, Lengths: $v")
    end

    for (k, v) in star_len_dict
        println("Time: $k, Lengths: $v")
    end
    =#
    X = []
    Y = []

    Y_tie = []
    Y_n = []
    Y_size = []
    Y_deg = []
    Y_ratio = []
    Y_novelty = []
    Y_avg_size = []
    Y_frac = []
    Y_first_size = []
    Y_frac_cont = []

    count = 1
    
    for (k, v) in star_len_dict
        plot_dict = DataStructures.SortedDict()
        #println(k)
        X_current = []
        Y_current = []
        for l in v
            print(stdout, "Count: $count \r")
            flush(stdout)
            

            ts_to_average = []
            n_to_average = []
            size_to_average = []
            first_size_to_average = []
            deg_to_average = []
            ratio_to_average = []
            novelty_to_average = []
            avg_size_to_average = []
            frac_to_average = []

            for (ego, info) in global_sorted_dict
                if info[7] == l
                    #println("Length: $l")
                    egonet = info[3]
                    star_count_dict = DataStructures.SortedDict()
                    
                    data_tie_strength, data_n, data_size, data_degrees, data_novelty, count_, final_index, avg_novelty_rate, avg_size, 
                        frac_simplices_before_ego = set_data_ordinal_time_no_user_set_length(egonet, ego, 0, 0, star_count_dict, k)
                    
                    merge!(+, plot_dict, data_size)
                    
                    
                end
            end
            count += 1
        end

        for (k, n) in plot_dict
            n = n / length(v)
            push!(X_current, k)
            push!(Y_current, n)
            # APPEND ARRAY TO X AND Y
            #println("Time: $k, Novelty: $n")
        end

        push!(X, X_current)
        push!(Y, Y_current)

    end
    # "500-1000" "1000-1500" "1500-2000" "2000-3000" "3000-4000" "4000+"
    Plots.plot(X, Y, xlabel = "ordinal time", ylabel = "incoming simplex size", label = ["50-100" "100-150" "150-200" "200-300" "300-500"], left_margin = 10mm, bottom_margin = 10mm, legend =:topright, title = "Incoming simplex size - Star", linewidth=0.5)
    Plots.savefig("$(dataset)_star_size.pdf")
end

function average_lengths(dataset::String)
    global_sorted_dict = load_sorted_dicts(dataset, true)
    def_lengths = []
    cont_lengths = []
    star_lengths = []
    for (ego, info) in global_sorted_dict
        print(stdout, "Ego: $ego \r")
        flush(stdout)

        def_egonet = info[1]
        cont_egonet = info[2]
        star_egonet = info[3]

        def_length = get_length(def_egonet)
        cont_length = get_length(cont_egonet)
        star_length = get_length(star_egonet)

        push!(def_lengths, def_length)
        push!(cont_lengths, cont_length)
        push!(star_lengths, star_length)
    end

    mean_def_length = mean(def_lengths)
    mean_cont_length = mean(cont_lengths)
    mean_star_length = mean(star_lengths)

    println("Average length of default egonets: $mean_def_length")
    println("Average length of contracted egonets: $mean_cont_length")
    println("Average length of star egonets: $mean_star_length")

    return mean_def_length, mean_cont_length, mean_star_length
end

function plot_lengths(dataset::String, log::Bool = false)
    global_sorted_dict = load_sorted_dicts(dataset, true)

    star_lengths = [info[7] for (ego, info) in global_sorted_dict]
    sort!(star_lengths)
    unique!(star_lengths)
    #println(star_lengths)
    
    def_lengths = []
    cont_lengths = []
    star_counts = []
    for length in star_lengths
        def_lens_to_average = []
        cont_lens_to_average = []
        count = 0
        for (ego, info) in global_sorted_dict
            if info[7] == length
                push!(def_lens_to_average, info[5])
                push!(cont_lens_to_average, info[6])
                count += 1
            end
        end
        push!(star_counts, count)
        push!(def_lengths, mean(def_lens_to_average))
        push!(cont_lengths, mean(cont_lens_to_average))
    end

    X = star_lengths
    Y = star_counts

    if log
        Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of star egonet x", ylabel = "number of egonets of length x", scale=:log10, left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "range of lengths of star egos", linewidth=0.5)
        Plots.savefig("$(dataset)_star_range_of_lengths.pdf")
    else
        Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of star egonet x", ylabel = "number of egonets of length x", left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "range of lengths of star egos", linewidth=0.5)
        Plots.savefig("$(dataset)_star_range_of_lengths.pdf")
    end

    Y = def_lengths
        
    Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of star egonet", ylabel = "length of corresponding default egonet", left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "star_length_vs_def_length", linewidth=0.5)
    Plots.savefig("$(dataset)_star_length_vs_def_length.pdf")

    Y = cont_lengths

    Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of star egonet", ylabel = "length of corresponding contracted egonet", left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "star_length_vs_cont_length", linewidth=0.5)
    Plots.savefig("$(dataset)_star_length_vs_cont_length.pdf")

    def_lengths = [info[5] for (ego, info) in global_sorted_dict]
    sort!(def_lengths)
    unique!(def_lengths)
    cont_lengths = [info[6] for (ego, info) in global_sorted_dict]
    sort!(cont_lengths)
    unique!(cont_lengths)
    def_counts = []
    cont_counts = []

    for length in def_lengths
        count = 0
        for (ego, info) in global_sorted_dict
            if info[5] == length
                count+=1
            end
        end
        push!(def_counts, count)
    end

    for length in cont_lengths
        count = 0
        for (ego, info) in global_sorted_dict
            if info[6] == length
                count+=1
            end
        end
        push!(cont_counts, count)
    end
    
    X = def_lengths
    Y = def_counts

    if log
        Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of def egonet x", ylabel = "number of egonets of length x", scale=:log10, left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "range of lengths of default egos", linewidth=0.5)
        Plots.savefig("$(dataset)_def_range_of_lengths.pdf")
    else
        Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of def egonet x", ylabel = "number of egonets of length x", xlim=(0, 50), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "range of lengths of default egos", linewidth=0.5)
        Plots.savefig("$(dataset)_def_range_of_lengths.pdf")
    end

    X = cont_lengths
    Y = cont_counts

    if log
        Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of cont egonet x", ylabel = "number of egonets of length x", scale=:log10, left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "range of lengths of contracted egos", linewidth=0.5)
        Plots.savefig("$(dataset)_cont_range_of_lengths.pdf")
    else
        Plots.plot(X, Y, seriestype = :scatter, xlabel = "length of cont egonet x", ylabel = "number of egonets of length x", xlim=(0, 100), left_margin = 10mm, bottom_margin = 10mm, legend = false, title = "range of lengths of contracted egos", linewidth=0.5)
        Plots.savefig("$(dataset)_cont_range_of_lengths.pdf")
    end

end

function load_egos_with_data(dataset::String, num_egos_limit::Int64, def_length_bounds::Array{Int64, 1}, cont_length_bounds::Array{Int64, 1},
    star_length_bounds::Array{Int64, 1}, degree_bounds::Array{Int64, 1}, all::Bool = false)

    def_egos_to_return = []
    cont_egos_to_return = []
    star_egos_to_return = []
    
    def_count = 0
    cont_count = 0
    star_count = 0

    global_sorted_dict = load_sorted_dicts(dataset, true)

    if all == true
        return global_sorted_dict, collect(keys(global_sorted_dict)), collect(keys(global_sorted_dict)), collect(keys(global_sorted_dict))
    end

    for (ego, info) in global_sorted_dict
        if degree_bounds[1] <= info[4] <= degree_bounds[2]
            if def_length_bounds[1] <= info[5] <= def_length_bounds[2]
                if def_count < num_egos_limit
                    push!(def_egos_to_return, deepcopy(ego))
                    def_count += 1
                end
            end
            if cont_length_bounds[1] <= info[6] <= cont_length_bounds[2]
                if cont_count < num_egos_limit
                    push!(cont_egos_to_return, deepcopy(ego))
                    cont_count += 1
                end
            end
            if star_length_bounds[1] <= info[7] <= star_length_bounds[2]
                if star_count < num_egos_limit
                    push!(star_egos_to_return, deepcopy(ego))
                    star_count += 1
                end
            end
        end
    end

    # global_sorted_dict_with_data[ego] = [def_sorted_data, contracted_sorted_data, star_sorted_data, degree, def_len ,cont_len, star_len]

    #println(def_egos_to_return)
    println("Number of def egos: $(length(def_egos_to_return))")
    #println(cont_egos_to_return)
    println("Number of contracted egos: $(length(cont_egos_to_return))")
    #println(star_egos_to_return)
    println("Number of star egos: $(length(star_egos_to_return))")

    return global_sorted_dict, def_egos_to_return, cont_egos_to_return, star_egos_to_return
end

function store_sorted_dicts(dataset_name::String, num_samples::Int64, all::Bool = false)
    #println("# threads available = $(Threads.nthreads())")
    global_sorted_dict = DataStructures.SortedDict()

    dataset = read_txt_data(dataset_name)
    sampled_egos, B, eligible_egos = get_sampled_egos(dataset, num_samples)

    if all
        sampled_egos = eligible_egos
    end

    count = 0 
    println("Getting data...")
    #=@time =#
    for ego in eligible_egos
        print(stdout, "Count: $count \r")
        flush(stdout)
        degree, def_len, cont_len, star_len, exp_len, min = get_egonet_data_dict(dataset, ego, B)
        global_sorted_dict[ego] = [degree, def_len, cont_len, star_len, exp_len]
        count += 1
        if count % 100000 == 0
            println("Saving sorted dicts...")
            FileIO.save("../../../data/clc348/all_$(dataset_name)_global_sorted_dict.jld2","global_sorted_dict",global_sorted_dict)
            println("Saved sorted dicts!")
        end
    end 
    println("Got data!")
    println("Saving sorted dicts...")
    FileIO.save("../../../data/clc348/all_$(dataset_name)_global_sorted_dict.jld2","global_sorted_dict",global_sorted_dict)
    println("Saved sorted dicts!")
end

function store_sorted_dicts_with_data(dataset_name::String, num_samples::Int64, all::Bool = false)
    #println("# threads available = $(Threads.nthreads())")
    global_sorted_dict_with_data = DataStructures.SortedDict()

    dataset = read_txt_data(dataset_name)
    sampled_egos, B, eligible_egos = get_sampled_egos(dataset, num_samples)

    if all
        sampled_egos = eligible_egos
    end

    count = 0
    println("Getting data...")
    #=@time =#
    for ego in eligible_egos
        println("Ego: $(ego)")
        def_sorted_data, contracted_sorted_data, star_sorted_data, degree, def_len ,cont_len, star_len = get_egonet_data_dict_with_data(dataset, ego, B)
        global_sorted_dict_with_data[ego] = [def_sorted_data, contracted_sorted_data, star_sorted_data, degree, def_len ,cont_len, star_len]
        count += 1
        if count % 100000 == 0
            println("Saving sorted dicts...")
            FileIO.save("../../../data/clc348/global_dicts/all_$(dataset_name)_global_sorted_dict_with_data.jld2","global_sorted_dict_with_data",global_sorted_dict_with_data)
            println("Saved sorted dicts!")
        end
    end 
    println("Got data!")
    println("Saving sorted dicts...")
    FileIO.save("../../../data/clc348/global_dicts/$(dataset_name)_global_sorted_dict_with_data.jld2","global_sorted_dict_with_data",global_sorted_dict_with_data)
    println("Saved sorted dicts!")
end

function store_all_sorted_dicts(dataset_names::Array{String, 1}, num_samples::Int64, all::Bool = false)
    for dataset_name in dataset_names
        store_sorted_dicts(dataset_name, num_samples, all)
    end
end

function load_sorted_dicts(dataset_name::String, with_data::Bool)
    println("Loading...")
    if with_data
        dict = FileIO.load("../../../data/clc348/global_dicts/all_$(dataset_name)_global_sorted_dict_with_data.jld2","global_sorted_dict_with_data")
    else
        dict = FileIO.load("../../../data/clc348/global_dicts/$(dataset_name)_global_sorted_dict.jld2","global_sorted_dict")
    end
    println("Loaded!")
    return dict
end

function set_globals()
    #TODO
end

function get_B1_matrix(dataset::HONData)
    A1, At1, B1 = basic_matrices(dataset.simplices, dataset.nverts)
    return B1
end

function egonet_stats(dataset_name::String, num_egos_limit::Int64, 
    def_length_bounds::Array{Int64, 1}, cont_length_bounds::Array{Int64, 1},
    star_length_bounds::Array{Int64, 1},
    degree_bounds::Array{Int64, 1}, degrees_in_percentile::Bool, with_data::Bool, all::Bool = false)
    println("Dataset = $dataset_name")
    
    # Init all necessary variables and data containers
    default_total_dict_tie, default_total_dict_n, default_total_dict_size, default_total_dict_degrees, default_total_dict_int_size = 
        [DataStructures.SortedDict() for _ = 1:5]
    proj_total_dict_tie, proj_total_dict_n, proj_total_dict_size, proj_total_dict_degrees, proj_total_dict_int_size = 
        [DataStructures.SortedDict() for _ = 1:5]
    star_total_dict_tie, star_total_dict_n, star_total_dict_size, star_total_dict_degrees, star_total_dict_int_size = 
        [DataStructures.SortedDict() for _ = 1:5]
    not_def_total_dict_tie, not_def_total_dict_n, not_def_total_dict_size, not_def_total_dict_degrees, not_def_total_dict_int_size = 
        [DataStructures.SortedDict() for _ = 1:5]

    default_total_dict_tie_no_user, default_total_dict_n_no_user, default_total_dict_size_no_user, default_total_dict_degrees_no_user, default_total_dict_int_size_no_user = 
        [DataStructures.SortedDict() for _ = 1:5]
    proj_total_dict_tie_no_user, proj_total_dict_n_no_user, proj_total_dict_size_no_user, proj_total_dict_degrees_no_user, proj_total_dict_int_size_no_user = 
        [DataStructures.SortedDict() for _ = 1:5]
    star_total_dict_tie_no_user, star_total_dict_n_no_user, star_total_dict_size_no_user, star_total_dict_degrees_no_user, star_total_dict_int_size_no_user = 
        [DataStructures.SortedDict() for _ = 1:5]
    not_def_total_dict_tie_no_user, not_def_total_dict_n_no_user, not_def_total_dict_size_no_user, not_def_total_dict_degrees_no_user, not_def_total_dict_int_size_no_user = 
        [DataStructures.SortedDict() for _ = 1:5]

    default_count_list = []
    default_final_list = []
    proj_count_list = []
    proj_final_list = []
    star_count_list = []
    star_final_list = []
    not_def_count_list = []
    not_def_final_list = []
    #number_sampled_egos = length(sampled_egos)
    println("# threads available = $(Threads.nthreads())")
    count_lists = [default_count_list, proj_count_list, star_count_list, not_def_count_list]
    final_lists = [default_final_list, proj_final_list, star_final_list, not_def_final_list]

    default_count_dict, proj_count_dict, star_count_dict, not_def_count_dict = [DataStructures.SortedDict() for _ = 1:4]
    count_list = [default_count_dict, proj_count_dict, star_count_dict, not_def_count_dict]

    default_count_dict_no_user, proj_count_dict_no_user, star_count_dict_no_user, not_def_count_dict_no_user = 
        [DataStructures.SortedDict() for _ = 1:4]
    count_list_no_user = [default_count_dict_no_user, proj_count_dict_no_user, star_count_dict_no_user, not_def_count_dict_no_user]

    default_list = [default_total_dict_tie, default_total_dict_n, default_total_dict_size, 
        default_total_dict_degrees, default_total_dict_int_size]
    proj_list = [proj_total_dict_tie, proj_total_dict_n, proj_total_dict_size, 
        proj_total_dict_degrees, proj_total_dict_int_size]
    star_list = [star_total_dict_tie, star_total_dict_n, star_total_dict_size, 
        star_total_dict_degrees, star_total_dict_int_size]
    not_def_list = [not_def_total_dict_tie, not_def_total_dict_n, not_def_total_dict_size, 
        not_def_total_dict_degrees, not_def_total_dict_int_size]
    total_dict_lists = [default_list, proj_list, star_list, not_def_list]

    default_list_no_user = [default_total_dict_tie_no_user, default_total_dict_n_no_user, default_total_dict_size_no_user, 
        default_total_dict_degrees_no_user, default_total_dict_int_size_no_user]
    proj_list_no_user = [proj_total_dict_tie_no_user, proj_total_dict_n_no_user, proj_total_dict_size_no_user, 
        proj_total_dict_degrees_no_user, proj_total_dict_int_size_no_user]
    star_list_no_user = [star_total_dict_tie_no_user, star_total_dict_n_no_user, star_total_dict_size_no_user, 
        star_total_dict_degrees_no_user, star_total_dict_int_size_no_user]
    not_def_list_no_user = [not_def_total_dict_tie_no_user, not_def_total_dict_n_no_user, not_def_total_dict_size_no_user, 
        not_def_total_dict_degrees_no_user, not_def_total_dict_int_size_no_user]
    total_dict_lists_no_user = [default_list_no_user, proj_list_no_user, star_list_no_user, not_def_list_no_user]

    # Compile global hypergraph degree of each node, have to go through all egos here because you need a gloab degree before next steps
    #global_degree_dict, low_percentile, high_percentile = get_global_degrees(dataset, sampled_egos, B1)
    #low = 10
    #high = 100
    #println("Global max time = $global_max_time")

    dataset = read_txt_data(dataset_name)
    B1 = get_B1_matrix(dataset)
    #=
    if degrees_in_percentile
        def_egos, cont_egos, star_egos, exp_egos = load_egos_percentile(dataset_name, num_egos_limit ,def_length_bounds, cont_length_bounds, 
        star_length_bounds, exp_length_bounds, degree_bounds, all)
    else
        def_egos, cont_egos, star_egos = load_egos(dataset_name, num_egos_limit ,def_length_bounds, cont_length_bounds, 
        star_length_bounds, degree_bounds, all)
    end
    =#

    global_sorted_dict, def_egos, cont_egos, star_egos = load_egos_with_data(dataset_name, num_egos_limit, def_length_bounds, cont_length_bounds, star_length_bounds, degree_bounds)

    num_sampled_egos = 0

    egos = [def_egos, cont_egos, star_egos]
    upper_bounds = [def_length_bounds[2], cont_length_bounds[2], star_length_bounds[2]]
    lengths = [length(def_egos), length(cont_egos), length(star_egos)]
    #return
    #println(lengths)
    # Get all relavent statistics and store in the above dictionaries (per ego)
    #=Threads.@threads =#for i in 1:3 #
        println("Current ego type: $(ego_types[i])")
        for (ego_num, ego) in enumerate(egos[i])
            print(stdout, "$ego_num \r")
            flush(stdout)
            #println()
            #println("$ego_num, $ego")
            #println()
            
            if with_data # (& IF PROJECTED)
                get_ego_data_with_data_projected(global_sorted_dict, ego, B1, ego_types, count_lists, final_lists, total_dict_lists, 
                    total_dict_lists_no_user, 1, count_list, count_list_no_user, upper_bounds[i], i)
                #get_ego_data_with_data(global_sorted_dict, ego, B1, ego_types, count_lists, final_lists, total_dict_lists, 
                #    total_dict_lists_no_user, 1, count_list, count_list_no_user, upper_bounds[i], i)
            else
                get_ego_data(dataset, ego, B1, ego_types, count_lists, final_lists, total_dict_lists, 
                    total_dict_lists_no_user, 1, count_list, count_list_no_user, upper_bounds[i], i)
            end
            num_sampled_egos += 1
                
        end
        #println()
    end
    #=
    println()
    for (k, v) in default_total_dict_n
        println("Time = $k, Value = $v")
    end
    println()
    for (k, v) in default_total_dict_n_no_user
        println("Time = $k, Value = $v")
    end
    println()
    for (k, v) in proj_total_dict_n
        println("Time = $k, Value = $v")
    end
    println()
    for (k, v) in proj_total_dict_n_no_user
        println("Time = $k, Value = $v")
    end
    println()
    for (k, v) in star_total_dict_n
        println("Time = $k, Value = $v")
    end
    println()
    for (k, v) in star_total_dict_n_no_user
        println("Time = $k, Value = $v")
    end
    return
    =#
    #return
    # Plot dictionaries
    for i in 1:3
        plot_final_dicts(ego_types[i], dataset_name, total_dict_lists[i], total_dict_lists_no_user[i], upper_bounds[i], final_lists[i], lengths[i], count_list[i], count_list_no_user[i])
    end

    println("DONE! Number of actual egos sampled = $(num_sampled_egos)")
    return
end

# Construct SortedDict (ego->degree) for a given set of sampled egos
function get_global_degrees(dataset::HONData, sampled_egos::Array{Int64, 1}, B::SpIntMat)
    global_degree_dict = DataStructures.SortedDict()
    global_max_time = 0
    #global_min_time = 2^50
    for ego in sampled_egos
        max_time = 0
        in_egonet = zeros(Bool, size(B, 1))
        in_egonet[ego] = true
        in_egonet[findnz(B[:, ego])[1]] .= true
        degrees = 0
        curr_ind = 1

        for (nvert, time) in zip(dataset.nverts, dataset.times)
            end_ind = curr_ind + nvert - 1
            simplex = dataset.simplices[curr_ind:end_ind]
            curr_ind += nvert
            simplex_in_egonet = [v for v in simplex if in_egonet[v]]
            
            #global_min_time = min(global_min_time, time)
            #global_max_time = max(global_max_time, time)
            if length(simplex_in_egonet) > 1
                degrees+=1
                max_time+=1
            end
        end
        global_degree_dict[ego] = degrees
        global_max_time = max(global_max_time, max_time)
    end

    #max_time = global_max_time - global_min_time

    low_percentile = percentile(values(global_degree_dict), 0)
    high_percentile = percentile(values(global_degree_dict), 100)
    
    println("75 percentile = $(low_percentile)")
    println("90 percentile = $(percentile(values(global_degree_dict), 90))")
    println("100 percentile = $(high_percentile)")
    #=
    #println("Degree of ego 1 = $(global_degree_dict[1])")
    for (k, v) in global_degree_dict
        println("Key = $k, Value = $v")
    end
    println()
    =#
    return global_degree_dict, low_percentile, high_percentile #, global_max_time
end

function update_totals(current_array::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}}, 
    total_array::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}})
    for i = 1:5
        merge!(+, total_array[i], current_array[i])
    end
end

function get_ego_data(dataset::HONData, ego::Int64, B1::SpIntMat, ego_types::Array{String,1}, 
    count_lists::Array{Array{Any,1},1}, final_lists::Array{Array{Any,1},1}, 
    total_dict_lists::Array{Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},1}, 
    total_dict_lists_no_user::Array{Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},1}, 
    step::Int64,
    count_list::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},
    count_list_no_user::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},
    global_max_time::Int64, i::Int64)
    
    egonet, global_min = get_egonet_data_dict_two(dataset, ego, B1, ego_types[i])

    data_tie_strength, data_n, data_size, data_degrees, data_int_size, count, final_index = 
        set_data_ordinal_time(egonet, global_min, step, count_list[i], global_max_time)

    data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user, count_no_user, final_index_no_user = 
        set_data_ordinal_time_no_user(egonet, ego, global_min, step, count_list_no_user[i], global_max_time)

    push!(count_lists[i], count)
    push!(final_lists[i], final_index)

    current_array = [data_tie_strength, data_n, data_size, data_degrees, data_int_size]
    total_array = total_dict_lists[i]
    update_totals(current_array, total_array)

    current_array_no_user = [data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user]
    total_array_no_user = total_dict_lists_no_user[i]
    update_totals(current_array_no_user, total_array_no_user)
    
    #=
    Threads.@threads for i = 1:4
        egonet, global_min = get_egonet_data_dict(dataset, ego, B1, ego_types[i])

        data_tie_strength, data_n, data_size, data_degrees, data_int_size, count, final_index = 
            set_data_ordinal_time(egonet, global_min, step, count_list[i], global_max_time)

        data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user, count_no_user, final_index_no_user = 
            set_data_ordinal_time_no_user(egonet, ego, global_min, step, count_list_no_user[i], global_max_time)

        push!(count_lists[i], count)
        push!(final_lists[i], final_index)
        #=
        count_list[i] = count_dict
        count_list_no_user[i] = count_dict_no_user
        =#
        #=
        if i == 1
            println("Ego:")
            for (k, v) in egonet
                println("Key = $k, Value = $v")
            end

            println("Count dict:")
            for (k, v) in count_list[i]
                println("Key = $k, Value = $v")
            end

            println("Count dict no user:")
            for (k, v) in count_list_no_user[i]
                println("Key = $k, Value = $v")
            end
        end
        =#
        current_array = [data_tie_strength, data_n, data_size, data_degrees, data_int_size]
        total_array = total_dict_lists[i]
        update_totals(current_array, total_array)

        current_array_no_user = [data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user]
        total_array_no_user = total_dict_lists_no_user[i]
        update_totals(current_array_no_user, total_array_no_user)
    end
    =#
end

function get_ego_data_with_data(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64, B1::SpIntMat, ego_types::Array{String,1}, 
    count_lists::Array{Array{Any,1},1}, final_lists::Array{Array{Any,1},1}, 
    total_dict_lists::Array{Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},1}, 
    total_dict_lists_no_user::Array{Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},1}, 
    step::Int64,
    count_list::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},
    count_list_no_user::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},
    global_max_time::Int64, i::Int64)
    
    egonet = global_sorted_dict[ego][i]
    global_min = 0

    data_tie_strength, data_n, data_size, data_degrees, data_int_size, count, final_index = 
        set_data_ordinal_time(egonet, global_min, step, count_list[i], global_max_time)

    data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user, count_no_user, final_index_no_user = 
        set_data_ordinal_time_no_user(egonet, ego, global_min, step, count_list_no_user[i], global_max_time)

    push!(count_lists[i], count)
    push!(final_lists[i], final_index)

    current_array = [data_tie_strength, data_n, data_size, data_degrees, data_int_size]
    total_array = total_dict_lists[i]
    update_totals(current_array, total_array)

    current_array_no_user = [data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user]
    total_array_no_user = total_dict_lists_no_user[i]
    update_totals(current_array_no_user, total_array_no_user) 
end

function get_ego_data_with_data_projected(global_sorted_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64, B1::SpIntMat, ego_types::Array{String,1}, 
    count_lists::Array{Array{Any,1},1}, final_lists::Array{Array{Any,1},1}, 
    total_dict_lists::Array{Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},1}, 
    total_dict_lists_no_user::Array{Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},1}, 
    step::Int64,
    count_list::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},
    count_list_no_user::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1},
    global_max_time::Int64, i::Int64)
    
    egonet = global_sorted_dict[ego][i]
    data = absolute_to_projected(egonet, ego)
    global_min = 0

    data_tie_strength, data_n, data_size, data_degrees, data_int_size, count, final_index = 
        set_data_ordinal_time(data, global_min, step, count_list[i], global_max_time)

    data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user, count_no_user, final_index_no_user = 
        set_data_ordinal_time_no_user(data, ego, global_min, step, count_list_no_user[i], global_max_time)

    push!(count_lists[i], count)
    push!(final_lists[i], final_index)

    current_array = [data_tie_strength, data_n, data_size, data_degrees, data_int_size]
    total_array = total_dict_lists[i]
    update_totals(current_array, total_array)

    current_array_no_user = [data_tie_strength_no_user, data_n_no_user, data_size_no_user, data_degrees_no_user, data_int_size_no_user]
    total_array_no_user = total_dict_lists_no_user[i]
    update_totals(current_array_no_user, total_array_no_user) 
end

function plot_final_dicts(ego_type::String, dataset_name::String, 
    list::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1}, 
    list_no_user::Array{DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},1}, 
    len::Int64, final_lists::Array{Any,1}, num_sampled_egos::Int64,
    count_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},
    count_dict_no_user::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})

    final_dict_tie, final_dict_n, final_dict_size, final_dict_degrees, final_dict_int_size = 
        [DataStructures.SortedDict() for _ = 1:5]

    final_dict_tie_no_user, final_dict_n_no_user, final_dict_size_no_user, final_dict_degrees_no_user, final_dict_int_size_no_user = 
        [DataStructures.SortedDict() for _ = 1:5]

    final_dicts = [final_dict_tie, final_dict_n, final_dict_size, final_dict_degrees, final_dict_int_size]
    final_dicts_no_user = [final_dict_tie_no_user, final_dict_n_no_user, final_dict_size_no_user, final_dict_degrees_no_user, final_dict_int_size_no_user]
    data_type = ["tie", "n", "size", "degrees", "novelty"]
    #println(num_sampled_egos)
    #x_lim = sum(final_lists) / length(final_lists)
    # plot_final_dicts(ego_types[i], dataset_name, total_dict_lists[i], total_dict_lists_no_user[i], number_sampled_egos, final_lists[i], num_sampled_egos)
    # total_dict_lists = [default_list, proj_list, star_list, not_def_list], each is a list
    # default_list = [default_total_dict_tie, default_total_dict_n, default_total_dict_size, default_total_dict_degrees, default_total_dict_int_size]
    for i in 1:5

        for (key, value) in list[i]
            final_dicts[i][key] = (list[i][key]) / num_sampled_egos # count_dict[key]
            # final_dict_tie[key] = default_total_dict_tie[key] / default_dict[key]
            #println("Length = $(length(list[i][key]))")
        end
        #println(length(final_dicts[i]))
        for (key, value) in list_no_user[i]
            final_dicts_no_user[i][key] = (list_no_user[i][key]) / num_sampled_egos # count_dict_no_user[key]
        end

        ego_plot(final_dicts[i], final_dicts_no_user[i], len, dataset_name, data_type[i], "$(ego_type)")
    end
end

function ego_plot(final_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, 
    final_dict_no_user::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, 
    l::Int64, dataset_name::String, option::String, ego_type::String)
    plot_params = all_datasets_params() 
    X_user = []
    Y_user = []
    X_no_user = []
    Y_no_user = []
    #println("final_dict")
    for (key, value) in final_dict
        append!(X_user, key)
        append!(Y_user, value)
        #println("Key = $(key), Value = $(value)")
    end

    #println("final_dict_no_user")
    for (key, value) in final_dict_no_user
        append!(X_no_user, key)
        append!(Y_no_user, value)
        #println("Key = $(key), Value = $(value)")
    end
    println()
    #X = [X_user, X_no_user]
    #Y = [Y_user, Y_no_user]
    X = X_no_user
    Y = Y_no_user
    subplot(221)
    #println("limit = $(limit)")
    label = ""
    if option == "tie"
        label = "average tie strength of egonet"
    elseif option == "n"
        label = "number of nodes in egonet"
    elseif option == "size"
        label = "average size of incoming simplex"
    elseif option == "degree"
        label = "average degree"
    elseif option == "novelty"
        label = "average novelty"
    end
    #FileIO.save("../../../data/clc348/set_length/absolute/$(ego_type)_$(dataset_name)_$(l)_$(option)_X.jld2","X",X)
    #FileIO.save("../../../data/clc348/set_length/absolute/$(ego_type)_$(dataset_name)_$(l)_$(option)_Y.jld2","Y",Y)
    FileIO.save("../../../data/clc348/set_length/projected/$(ego_type)_$(dataset_name)_$(l)_$(option)_X.jld2","X",X)
    FileIO.save("../../../data/clc348/set_length/projected/$(ego_type)_$(dataset_name)_$(l)_$(option)_Y.jld2","Y",Y)
    return
    Plots.plot(X, Y, xlabel = "time relative to first simplex", ylabel = label, left_margin = 10mm, bottom_margin = 10mm, xlim=(1, l), legend = false, title = "$(ego_type)_$(dataset_name)_$(option)", linewidth=0.5)

    Plots.savefig("$(ego_type)_$(dataset_name)_$(option).pdf")

    #return
end

function egonet_stats_individual(dataset_name::String, num_egos::Int64)
    # read data
    dataset = read_txt_data(dataset_name)
    A1, At1, B1 = basic_matrices(dataset.simplices, dataset.nverts)

    # Get eligible egos
    n = size(B1, 1)
    tri_order = proj_graph_degree_order(B1)
    in_tri = zeros(Int64, n, Threads.nthreads())
    Threads.@threads for i = 1:n
        for (j, k) in neighbor_pairs(B1, tri_order, i)
            if B1[j, k] > 0
                tid = Threads.threadid()
                in_tri[[i, j, k], tid] .= 1
            end
        end
    end
    eligible_egos = findall(vec(sum(in_tri, dims=2)) .> 0)
    num_eligible = length(eligible_egos)
    println("$num_eligible eligible egos")

    # Sample from eligible egos
    sampled_egos =
        eligible_egos[StatsBase.sample(1:length(eligible_egos),
                                       num_egos, replace=false)]

    function get_base_data(dataset::HONData)
        sorted_data = DataStructures.SortedDict()
        let curr_ind = 0
        for (nvert, time) in zip(dataset.nverts, dataset.times)
        simplex = dataset.simplices[(curr_ind + 1):(curr_ind + nvert)]
            curr_ind += nvert
            sorted_data[time] = []
        end
        end
        return sorted_data
    end

    base = get_base_data(dataset)

    # Collect statistics
    X = zeros(Float64, NUM_FEATS, length(sampled_egos))
    println("$sampled_egos sampled egos")
    
    X_n_final = []
    Y_n_final = []
    X_tie_final = []
    Y_tie_final = []
    X_size_final = []
    Y_size_final = []

    for (j, ego) in enumerate(sampled_egos)
        print(stdout, "$j \r")
        flush(stdout)
        egonet = egonet_dataset(dataset, ego, B1)
        A, At, B = basic_matrices(egonet.simplices, egonet.nverts)

        plot_params = all_datasets_params()
        X_tie = []
        Y_tie = []
        X_n = []
        Y_n = []
        X_size = []
        Y_size= []

	    data_tie_strength, data_n, data_size = set_data_individual(egonet)

        for (key, value) in data_tie_strength
            append!(X_tie, key)
            append!(Y_tie, value)
        end

        for (key, value) in data_n
            append!(X_n, key)
            append!(Y_n, value)
        end

        for (key, value) in data_size
            append!(X_size, key)
            append!(Y_size, value)
        end
        
	    push!(X_n_final, X_n)
        push!(Y_n_final, Y_n)
        
	    push!(X_tie_final, X_tie)
        push!(Y_tie_final, Y_tie)
        
	    push!(X_size_final, X_size)
        push!(Y_size_final, Y_size)

    end

    Plots.plot(X_n_final,Y_n_final, xlabel = "time relative to first simplex", ylabel = "number of nodes in egonet", ylim = (0, 200), left_margin = 10mm, bottom_margin = 10mm, title = "$(dataset_name)_all_samples_n", legend = false)
    Plots.savefig("$(dataset_name)_all_samples_n.pdf")

    Plots.plot(X_tie_final,Y_tie_final, xlabel = "time relative to first simplex", ylabel = "average tie strength of egonet", left_margin = 10mm, bottom_margin = 10mm, title = "$(dataset_name)_all_tie_n", legend = false)
    Plots.savefig("$(dataset_name)_all_samples_tie.pdf")

    Plots.plot(X_size_final,Y_size_final, xlabel = "time relative to first simplex", ylabel = "size of incoming simplex", left_margin = 10mm, bottom_margin = 10mm, title = "$(dataset_name)_all_samples_size", legend = false)
    Plots.savefig("$(dataset_name)_all_samples_size.pdf")

    return
end

function collect_egonet_data(num_egos::Int64, trial::Int64)
    Random.seed!(1234 * trial)  # reproducibility
    dataset_names = [row[1] for row in all_datasets_params()]
    ndatasets = length(dataset_names)
    X = zeros(Float64, 0, NUM_FEATS)
    labels = Int64[]
    for (ind, dname) in enumerate(dataset_names)
        println("$dname...")
        label = nothing
        if     (dname == "coauth-DBLP" ||
                dname == "coauth-MAG-Geology" ||
                dname == "coauth-MAG-History");      label = 0;
        elseif (dname == "tags-stack-overflow" ||
                dname == "tags-math-sx"        ||
                dname == "tags-ask-ubuntu");         label = 1;
        elseif (dname == "threads-stack-overflow" ||
                dname == "threads-math-sx"        ||
                dname == "threads-ask-ubuntu");      label = 2;
        elseif (dname == "contact-high-school" ||
                dname == "contact-primary-school");  label = 3;
        elseif (dname == "email-Eu" ||
                dname == "email-Enron");             label = 4;
        end
        if label != nothing
            X = [X; egonet_stats(dname, num_egos)]
            append!(labels, ones(Int64, num_egos) * label)
        end
    end
    save("output/egonets/egonet-data-$trial.jld2",
         Dict("X" => X, "labels" => labels))
end

function egonet_predict(feat_cols::Vector{Int64})
    accs_mlr = Float64[]
    accs_rnd = Float64[]

    for trial in 1:20
        (X_train, X_test, y_train, y_test) = egonet_train_test_data(trial)[1:4]
        X_train = X_train[:, feat_cols]
        X_test = X_test[:, feat_cols]
        model = LogisticRegression(fit_intercept=true, multi_class="multinomial",
                                   C=10, solver="newton-cg", max_iter=10000)
        ScikitLearn.fit!(model, X_train, y_train)
        rand_prob =
            sum([(sum(y_train .== l) / length(y_train))^2 for l in unique(y_train)])
        push!(accs_mlr, ScikitLearn.score(model, X_test, y_test))
        push!(accs_rnd, rand_prob)
    end

    @printf("%0.2f +/- %0.2f\n", mean(accs_mlr), std(accs_mlr))
    @printf("%0.2f +/- %0.2f\n", mean(accs_rnd), std(accs_rnd))
end
