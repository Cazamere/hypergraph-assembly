using Base.Threads
using Combinatorics
using DelimitedFiles
using FileIO
using JLD2
using Random
using SparseArrays
using StatsBase
using Distributed

import DataStructures

const NUM_FEATS = 3
const LOG_AVE_DEG = 1
const LOG_DENSITY = 2
const FRAC_OPEN   = 3

function read_txt_data(dataset::String)
    function read(filename::String)
        ret = Int64[]
        open(filename) do f
            for line in eachline(f)
                push!(ret, parse(Int64, line))
            end
        end
        return ret
    end
    return HONData(read("data/$(dataset)/$(dataset)-simplices.txt"),
                   read("data/$(dataset)/$(dataset)-nverts.txt"),
                   read("data/$(dataset)/$(dataset)-times.txt"),
                   dataset)
end

function read_node_labels(dataset::String)
    labels = Vector{String}()
    open("data/$(dataset)/$(dataset)-node-labels.txt") do f
        for line in eachline(f)
            push!(labels, join(split(line)[2:end], " "))
        end
    end
    return labels
end

function read_simplex_labels(dataset::String)
    labels = Vector{String}()
    open("data/$(dataset)/$(dataset)-simplex-labels.txt") do f
        for line in eachline(f)
            push!(labels, join(split(line)[2:end], " "))
        end
    end
    return labels
end

function read_closure_stats(dataset::String, simplex_size::Int64, initial_cutoff::Int64=100)
    keys = []
    probs, nsamples, nclosed = Float64[], Int64[], Int64[]
    data = readdlm("output/$(simplex_size)-node-closures/$(dataset)-$(simplex_size)-node-closures.txt")
    if initial_cutoff < 100
        data = readdlm("output/$(simplex_size)-node-closures/$(dataset)-$(simplex_size)-node-closures-$(initial_cutoff).txt")
    end
    for row_ind in 1:size(data, 1)
        row = convert(Vector{Int64}, data[row_ind, :])
        push!(keys, tuple(row[1:simplex_size]...))
        push!(nsamples, row[end - 1])
        push!(nclosed, row[end])
    end
    return (keys, nsamples, nclosed)
end

function egonet_train_test_data(trial::Int64)
    Random.seed!(444)  # for reproducibility
    data = load("output/egonets/egonet-data-$trial.jld2")
    X = data["X"]
    y = data["labels"]
    yf = data["full_labels"]
    inds = randperm(length(y))
    X = X[inds, :]
    y = y[inds]
    yf = yf[inds]
    
    train_inds = Int64[]
    test_inds  = Int64[]
    for label in sort(unique(y))
        inds = findall(y .== label)
        end_ind = convert(Int64, round(length(inds) * 0.8))
        append!(train_inds, inds[1:end_ind])
        append!(test_inds,  inds[(end_ind + 1):end])
    end
    
    X_train, X_test = X[train_inds, :], X[test_inds, :]
    y_train, y_test = y[train_inds], y[test_inds]
    yf_train, yf_test = yf[train_inds], yf[test_inds]    
    return (X_train, X_test, y_train, y_test, yf_train, yf_test)
end

function absolute_to_projected(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64)
    other_dict = DataStructures.SortedDict()
    projected_dict = DataStructures.SortedDict()
    unique = []

    for (time, simplex_list) in egonet
        for simplex in simplex_list
            
            if ego in simplex
                for node in simplex
                    if !(node in unique)
                        push!(unique, node)
                    end
                end

                # push current simplex into projected_dict
                try
                    push!(projected_dict[time], deepcopy(simplex))
                catch e
                    projected_dict[time] = [deepcopy(simplex)]
                end

                # iterate through other_dict
                for (other_time, other_simplex_list) in other_dict
                    for other_simplex in other_simplex_list

                        if other_simplex == []
                            continue
                        end

                        all_unique = []
                        
                        # check if all nodes in other_simplex are in unique
                        for node in other_simplex
                            if (node in unique)
                                push!(all_unique, true)
                            else
                                push!(all_unique, false)
                            end   
                        end

                        if (other_time <= time) && all(all_unique)
                            push!(projected_dict[time], deepcopy(other_simplex))
                            empty!(other_simplex)
                        end
                    end
                    filter!(e->e!=[], other_simplex_list)
                end
            else
                all_unique = []
                        
                # check if all nodes in other_simplex are in unique
                for node in simplex
                    if (node in unique)
                        push!(all_unique, true)
                    else
                        push!(all_unique, false)
                    end   
                end

                if all(all_unique)
                    try
                        push!(projected_dict[time], deepcopy(simplex))
                    catch e
                        projected_dict[time] = [deepcopy(simplex)]
                    end
                else
                    try
                        push!(other_dict[time], deepcopy(simplex))
                    catch e
                        other_dict[time] = [deepcopy(simplex)]
                    end
                end
            end
        end
    end
    return projected_dict
end

function get_simplicies_until_user(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64)
    alter_simplicies = []
    first_user_simplex = []
    for (k, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            if ego in simplex
                user_simplicies, first_user_simplex = get_user_simplicies(egonet, ego)
                return alter_simplicies, user_simplicies, first_user_simplex
            else
                push!(alter_simplicies, simplex)
            end
        end

        
    end
end

function get_user_alter_simplicies(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64)
    user_simplicies = []
    alter_simplicies = []
    
    for (k, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            if ego in simplex
                push!(user_simplicies, simplex)
            else
                push!(alter_simplicies, simplex)
            end
        end

    end
    return user_simplicies, alter_simplicies
end

function get_all_simplicies(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    simplicies = []
    
    for (k, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            push!(simplicies, simplex)
        end

    end
    return simplicies
end

function get_user_simplicies(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64)
    simplicies = []
    
    for (k, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            if ego in simplex
                push!(simplicies, simplex)
            end
        end

    end
    #println("User Simplicies: $simplicies")
    first_user_simplex = simplicies[1]
    return simplicies, first_user_simplex
end

function get_alters(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64)
    alters = []
    for (k, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            for node in simplex
                if !(node in alters) && !(node == ego)
                    push!(alters, node)
                end
            end
        end
    end

    return alters
end

function get_alters_simplicies(simplicies::Array{Any, 1}, ego::Int64)
    alters = []

    for simplex in simplicies
        for node in simplex
            if !(node in alters) && !(node == ego)
                push!(alters, node)
            end
        end
    end

    return alters
end

function get_alternetwork_simplex(alternetwork::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, index::Int64)
    count = 1
    
    for (time, v) in alternetwork
        for simplex in v
            if count == index
                return simplex
            end
            count += 1
        end
    end
end

function get_alternetwork_n(alternetwork::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    unique = []
    alter_dict = DataStructures.SortedDict()
    count = 1

    for (time, v) in alternetwork
        for simplex in v
            for node in simplex
                if !(node in unique)
                    push!(unique, node)
                end
            end

            alter_dict[count] = length(unique)

            count += 1
        end
    end

    return alter_dict
end

function get_alternetwork_novelty(alternetwork::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    unique = []
    alter_dict = DataStructures.SortedDict()
    count = 1

    for (time, v) in alternetwork
        for simplex in v
            alter_dict[count] = length([n for n in simplex if !(n in unique)])

            for node in simplex
                if !(node in unique)
                    push!(unique, node)
                end
            end

            count += 1
        end
    end

    return alter_dict
end

function get_alternetwork(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, alter::Int64, ego::Int64)
    alternetwork = DataStructures.SortedDict()
    count = 0
    num_user_simplicies = 0
    time_until_alter = 1
    alter_arrived = false

    for (time, v) in egonet
        if v == []
            continue
        end
        #println("Time: $time, Simplex List: $v")
        for simplex in v
            if !(alter_arrived)
                time_until_alter += 1
            end
            if alter in simplex
                alter_arrived = true
                count += 1
                try
                    push!(alternetwork[time], deepcopy(simplex))
                catch e
                    alternetwork[time] = [deepcopy(simplex)]
                end
                if ego in simplex
                    num_user_simplicies += 1
                end
            end
        end
    end

    return alternetwork, count, num_user_simplicies, time_until_alter
end

function get_alternetwork_shallow_copy(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, alter::Int64, ego::Int64)
    alternetwork = DataStructures.SortedDict()
    count = 0
    num_user_simplicies = 0
    time_until_alter = 1
    alter_arrived = false

    for (time, v) in egonet
        if v == []
            continue
        end
        #println("Time: $time, Simplex List: $v")
        for simplex in v
            if !(alter_arrived)
                time_until_alter += 1
            end
            if alter in simplex
                alter_arrived = true
                count += 1
                try
                    push!(alternetwork[time], simplex)
                catch e
                    alternetwork[time] = [simplex]
                end
                if ego in simplex
                    num_user_simplicies += 1
                end
            end
        end
    end

    return alternetwork, count, num_user_simplicies, time_until_alter
end

function get_alternetwork_shallow_copy_simplicies(simplicies::Array{Any, 1}, alter::Int64, ego::Int64)
    alternetwork = []
    count = 0
    num_user_simplicies = 0
    time_until_alter = 1
    alter_arrived = false

    for simplex in simplicies
        if !(alter_arrived)
            time_until_alter += 1
        end
        if alter in simplex
            alter_arrived = true
            count += 1

            push!(alternetwork, simplex)

            if ego in simplex
                num_user_simplicies += 1
            end
        end
    end

    return alternetwork, count, num_user_simplicies, time_until_alter
end

function network_to_list(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    network_list = []
    
    for (time, v) in egonet
        if v == []
            continue
        end
        #println("Time: $time, Simplex List: $v")
        for simplex in v
            push!(network_list, simplex)
        end
    end

    return network_list
end

function predict_first_until_user_novelty(alter_simplicies::Array{Any, 1}, user_simplicies::Array{Any, 1}, ego::Int64)

    # Get unique array
    unique = []
    for simplex in alter_simplicies
        for node in simplex
            if !(node in unique)
                push!(unique, node)
            end
        end
    end

    highest_novelty_simplex = []
    max_novelty = 0
    for simplex in user_simplicies # replace with smallest simplicies out of all user_simplicies
        simplex_without_user = filter(e->e!=ego, simplex)
        novelty = length([v for v in simplex_without_user if !(v in unique)])

        if novelty > max_novelty
            highest_novelty_simplex = simplex
            max_novelty = novelty
        end
    end

    return highest_novelty_simplex
end

function predict_first_until_user_novelty_size(alter_simplicies::Array{Any, 1}, user_simplicies::Array{Any, 1}, ego::Int64)

    # Get unique array
    unique = []
    for simplex in alter_simplicies
        for node in simplex
            if !(node in unique)
                push!(unique, node)
            end
        end
    end

    highest_novelty_simplex = []
    max_novelty = 0

    # get all smallest simplicies
    min_length = minimum(length, user_simplicies)
    new_edges = filter(e->length(e)==min_length,user_simplicies)

    for simplex in new_edges
        simplex_without_user = filter(e->e!=ego, simplex)
        novelty = length([v for v in simplex_without_user if !(v in unique)])

        if novelty > max_novelty
            highest_novelty_simplex = simplex
            max_novelty = novelty
        end
    end

    return highest_novelty_simplex
end

function predict_first_until_user_degree_size(user_simplicies::Array{Any, 1}, all_simplicies::Array{Any, 1})
    min_length = minimum(length, user_simplicies)
    new_edges = filter(e->length(e)==min_length,user_simplicies)
    
    return predict_first_degree(new_edges, all_simplicies)
end

function predict_first_size(user_simplicies::Array{Any, 1})
    min_length = minimum(length, user_simplicies)
    new_edges = filter(e->length(e)==min_length,user_simplicies)
    return new_edges[rand(1:length(new_edges))]
end

function predict_first_degree(user_simplicies::Array{Any, 1}, all_simplicies::Array{Any, 1})
    degree_dict = DataStructures.SortedDict()
    for simplex in all_simplicies
        for node in simplex
            try
                degree_dict[node] += 1
            catch e
                degree_dict[node] = 1
            end
        end
    end
    max_degree_simplex = []
    max_degree = 0
    for simplex in user_simplicies
        degree_simp = mean([degree_dict[v] for v in simplex])
        if degree_simp > max_degree
            max_degree = degree_simp
            max_degree_simplex = simplex
        end
    end

    return max_degree_simplex
end

function get_degree_dict(simplicies::Array{Any, 1})
    degree_dict = DataStructures.SortedDict()

    for simplex in simplicies
        for node in simplex
            try
                degree_dict[node] += 1
            catch e
                degree_dict[node] = 1
            end
        end
    end

    return degree_dict
end

function get_all_degrees_star(user_simplicies::Array{Any, 1}, all_simplicies::Array{Any, 1})
    degree_dict = get_degree_dict(all_simplicies)
    
    max_degree_simplex = []
    max_degree = 0
    order_degree_dict = DataStructures.SortedDict()
    degrees = []

    for simplex in user_simplicies
        degree_simp = sum([degree_dict[v] for v in simplex]) - length(user_simplicies)
        #=
        if ego in simplex
            degree_simp -= length(user_simplicies)
        end
        =#
        push!(degrees, degree_simp)
    end

    return degrees
end

function get_top_degree(user_simplicies::Array{Any, 1}, all_simplicies::Array{Any, 1}, percent::Int64)
    degrees = get_all_degrees_star(user_simplicies, all_simplicies)

    min_degree = percentile(degrees, (100-percent))
    #println("min_degree: $min_degree")
    simps_to_return = []

    for simplex in user_simplicies
        degree_simp = mean([degree_dict[v] for v in simplex])
        if degree_simp >= min_degree
            push!(simps_to_return, simplex)
        end
    end

    return simps_to_return
end

function get_top_novelty(alter_simplicies::Array{Any, 1}, user_simplicies::Array{Any, 1}, ego::Int64, percent::Int64)
    # Get unique array
    unique = []
    for simplex in alter_simplicies
        for node in simplex
            if !(node in unique)
                push!(unique, node)
            end
        end
    end

    novelty_list = []
    max_novelty = 0
    for simplex in user_simplicies # replace with smallest simplicies out of all user_simplicies
        simplex_without_user = filter(e->e!=ego, simplex)
        novelty = length([v for v in simplex_without_user if !(v in unique)])
        push!(novelty_list, novelty)
    end

    min_novelty = percentile(novelty_list, 100 - percent)
    simps_to_return = []

    for simplex in user_simplicies
        simplex_without_user = filter(e->e!=ego, simplex)
        novelty = length([v for v in simplex_without_user if !(v in unique)])

        if novelty >= min_novelty
            push!(simps_to_return, simplex)
        end
    end

    return simps_to_return
end

function get_smallest_simplicies(simplicies::Array{Any, 1}, percent::Int64)
    lengths = []
    simplicies_to_return = []
    for simplex in simplicies
        push!(lengths, length(simplex))
    end

    for simplex in simplicies
        if length(simplex) < percentile(lengths, percent)
            push!(simplicies_to_return, simplex)
        end
    end

    return simplicies_to_return
end

function get_num_occurences(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, simplex_to_count::Array{Int64, 1})
    count = 0
    
    for (time, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            if simplex == simplex_to_count
                count += 1
            end
        end
    end

    return count
end

function get_length(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    count = 0
    
    for (time, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            count += 1
        end
    end

    return count
end

function get_sizes(simplicies::Array{Any, 1})
    return [length(s) for s in simplicies]
end

function get_max_mean_degree(simplicies::Array{Any, 1}, simplex_input::Array{Int64, 1}, ego::Int64)
    degree_dict = DataStructures.SortedDict()
    
    for simplex in simplicies
        for node in simplex
            try
                degree_dict[node] += 1
            catch e
                degree_dict[node] = 1
            end
        end
    end

    degree = sum([degree_dict[v] for v in simplex_input])

    if ego in simplex_input
        degree -= length(simplicies)
    end

    return degree
end

function get_isolated_nodes(simplicies::Array{Any, 1}, simplex::Array{Int64, 1})
    isolated_nodes = []

    simplicies_without_simplex = filter(e->e!=simplex, simplicies)

    nodes = get_nodes(simplicies_without_simplex)

    for node in simplex

        if node == -1
            continue
        end

        if !(node in nodes)
            push!(isolated_nodes, node)
        end
    end

    return isolated_nodes
end

function get_nodes(simplicies::Array{Any, 1})
    unique = []

    for simplex in simplicies
        for node in simplex
            if !(node in unique)
                push!(unique, node)
            end
        end
    end
    
    return unique
end

function get_similar_simplicies(simplicies::Array{Any, 1}, simplex::Array{Int64, 1})
    #println("Entered get_similar")
    similar_simplicies = []

    for s in simplicies
        #println("s (get): $s")
        if s == simplex
            #println("skipping")
            continue
        end

        intersection_size = length(intersect(s, simplex))

        if intersection_size > 1
            #println("Intersection! Pushing...")
            push!(similar_simplicies, s)
        end
    end
    #println("Leaving get_similar")
    return similar_simplicies
end

function get_distances(simplicies::Array{Any, 1}, simplex::Array{Int64, 1}, ss::Array{Int64, 1})
    distance = 0
    counting = false
    #println("In get_distances, simplicies: $simplicies")
    for s in simplicies

        if (simplex == ss)
            return 0
        end

        if (s == simplex || s == ss) && counting
            distance += 1
            return distance
        end

        if (s == simplex || s == ss) && counting == false
            counting = true
            continue
        end

        if counting
            distance += 1
        end
    end
end

function get_intersection_size_change(egonet::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})
    intersection_sizes = []
    skip = true
    prev = []
    for (time, v) in egonet
        if v == []
            continue
        end

        for simplex in v
            if skip
                prev = deepcopy(simplex)
                skip = false
                continue
            end

            intersection = intersect(prev, simplex)
            push!(intersection_sizes, intersection)
            prev = deepcopy(simplex)
        end
    end

    return intersection_sizes
end

function count_subsets(simplicies::Array{Any, 1}, simplex::Array{Int64, 1})
    count = 0
    #println("Subsets..., simplex: $simplex")
    for s in simplicies
        #println("s: $s")
        subset_check = [node for node in simplex if node in s]
        #println("subset_check: $subset_check")
        if (length(subset_check) == length(simplex))
            #println("subset!!")
            count += 1
        end
    end
    #println("Count: $count")
    return count - 1
end

function count_supersets(simplicies::Array{Any, 1}, simplex::Array{Int64, 1})
    count = 0

    for s in simplicies
        superset_check = [node for node in s if node in simplex]

        if (length(superset_check) == length(s))
            count += 1
        end
    end

    return count - 1
end

function set_data(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, global_min::Int64, step::Int64)
    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_int_size = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_int_size = DataStructures.SortedDict()
    end

    n = 0
    unique = []
    count = 0
    index = 0
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false

    #get mini value-TODO
    for (time, simplex) in data
        if simplex == []
            continue
        else
            mini = time
            break
        end
    end
    #println("Mini = $(mini)")
    #iterate through dataset
    for (time, simplex) in data
        if simplex == [] && start == false
            continue
        end
        start = true
        index = time - mini
        #println("Time = $(time), Index = $(index), Simplex = $(simplex)")
        if simplex == []
            data_tie_strength[index] = data_tie_strength[prev_index]
            #data_size[index] = 0
            data_num[index] = data_num[prev_index]
            data_degree[index] = data_degree[prev_index]
            #data_int_size[index] = data_int_size[prev_index]
            prev_index = index
            final_index = index
            continue
        end

        length_simplex = length(simplex)
        final_time = time
        push!(simplex_history, simplex)

        #calculate degrees per node for each iteration        
        for node in simplex
            if node in keys(degrees)
                degrees[node] += 1
            else
                degrees[node] = 1
            end
        end

        #calculate intersection size
        #=
        for s in simplex_history
            int_size_sum += length(intersect(s, simplex))
            if length(intersect(s, simplex)) > 0
                int_size_count += 1
            end
        end
        =#

        count += 1

        Threads.@threads for i = 1:2
            if i == 1
                combs = combinations(simplex)
                for x in combs
                    if length(x) != 2
                        continue
                    elseif get(pairs, sort(x), 0) == 0
                        pairs[sort(x)] = 1
                    else
                        pairs[sort(x)] += 1
                    end
                end
            elseif i == 2
                for x in simplex
                    if !(x in unique)
                        push!(unique, x)
                    end
                end
            end
        end

        length_unique = length(unique)
        n = length(keys(pairs))
        prev_index = index
        #println("time = $time; simplex = $simplex; n = $n; mini = $mini; index = $index, num = $(length(unique))")

        if (absolute) #absolute time as given in the dataset
            if (n == 0)
                data_tie_strength[time] = Float64(0)
                data_size[time] = length_simplex
                data_num[time] = length_unique
                data_degree[time] = Float64(1)
                data_int_size[time] = int_size_sum / int_size_count
                continue
            end

            data_tie_strength[time] = Float64(sum(values(pairs)) / n)
            data_num[time] = length_unique
            data_size[time] = length_simplex
            data_degree[time] = sum(values(degrees)) / length_unique
            #if (int_size_count == 0)
            #data_int_size[time] = Float64(int_size_sum / int_size_count)
        else #relative time (to first incoming simplex)
            if (n == 0)
                data_tie_strength[index] = Float64(0)
                data_size[index] = length_simplex
                data_num[index] = length_unique
                data_degree[index] = Float64(1) # sum(values(degrees))/ length_unique
                #data_int_size[index] = Float64(int_size_sum / int_size_count)
                data_int_size[index] = Float64(0)
                continue
            end

            data_tie_strength[index] = Float64(sum(values(pairs)) / n)
            data_num[index] = length_unique
            data_size[index] = length_simplex
            data_degree[index] = sum(values(degrees))/ length_unique
            #data_int_size[index] = Float64(int_size_sum / int_size_count)
        end
        final_index = index
        #=
        if index == 0 && !(absolute) && int_size_count == 0
            data_int_size[index] = 0
        end
        =#

        #clean up plots
    end
    for i in 1:step:(mini - global_min)
        data_tie_strength[final_index + i] = data_tie_strength[final_index]
        data_num[final_index + i] = data_num[final_index]
        data_degree[final_index + i] = data_degree[final_index]
        #data_int_size[index] = data_int_size[prev_index]
    end
    return data_tie_strength, data_num, data_size, data_degree, data_int_size, count, index
end

function set_data_no_user(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, ego::Int64, global_min::Int64, step::Int64)
    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_int_size = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_int_size = DataStructures.SortedDict()
    end

    n = 0
    unique = []
    count = 0
    index = 0
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false

    #get mini value-TODO
    for (time, simplex) in data
        if simplex == []
            continue
        else
            mini = time
            break
        end
    end

    #iterate through dataset
    for (time, simplex_full) in data
        if simplex_full == [] && start == false
            continue
        end
        start = true
        simplex = []
        for v in simplex_full
            if v != ego
                push!(simplex, v)
            end
        end
        index = time - mini
        if simplex == []
            if count == 0
                continue
            end
            data_tie_strength[index] = data_tie_strength[prev_index]
            data_size[index] = 0
            data_num[index] = data_num[prev_index]
            data_degree[index] = data_degree[prev_index]
            #data_int_size[index] = data_int_size[prev_index]
            prev_index = index
            final_index = index
            continue
        end

        length_simplex = length(simplex)
        final_time = time
        push!(simplex_history, simplex)
        #println("Simplex = $(simplex)")

        #calculate degrees per node for each iteration        
        for node in simplex
            if node in keys(degrees)
                degrees[node] += 1
            else
                degrees[node] = 1
            end
        end

        #calculate intersection size
        #=
        for s in simplex_history
            int_size_sum += length(intersect(s, simplex))
            if length(intersect(s, simplex)) > 0
                int_size_count += 1
            end
        end
        =#

        count += 1

        Threads.@threads for i = 1:2
            if i == 1
                combs = combinations(simplex)
                for x in combs
                    if length(x) != 2
                        continue
                    elseif get(pairs, sort(x), 0) == 0
                        pairs[sort(x)] = 1
                    else
                        pairs[sort(x)] += 1
                    end
                end
            elseif i == 2
                for x in simplex
                    if !(x in unique)
                        push!(unique, x)
                    end
                end
            end
        end

        length_unique = length(unique)
        n = length(keys(pairs))
        prev_index = index
        #println("time = $time; simplex = $simplex; n = $n; mini = $mini; index = $index, num = $(length(unique))")

        if (absolute) #absolute time as given in the dataset
            if (n == 0)
                data_tie_strength[time] = Float64(0)
                data_size[time] = length_simplex
                data_num[time] = length_unique
                data_degree[time] = Float64(1)
                data_int_size[time] = int_size_sum / int_size_count
                continue
            end

            data_tie_strength[time] = Float64(sum(values(pairs)) / n)
            data_num[time] = length_unique
            data_size[time] = length_simplex
            data_degree[time] = sum(values(degrees)) / length_unique
            #if (int_size_count == 0)
            #data_int_size[time] = Float64(int_size_sum / int_size_count)
        else #relative time (to first incoming simplex)
            if (n == 0)
                data_tie_strength[index] = Float64(0)
                data_size[index] = length_simplex
                data_num[index] = length_unique
                data_degree[index] = Float64(1) #sum(values(degrees))/ length_unique
                #data_int_size[index] = Float64(int_size_sum / int_size_count)
                data_int_size[index] = Float64(0)
                continue
            end

            data_tie_strength[index] = Float64(sum(values(pairs)) / n)
            data_num[index] = length_unique
            data_size[index] = length_simplex
            data_degree[index] = sum(values(degrees))/ length_unique
            #data_int_size[index] = Float64(int_size_sum / int_size_count)
        end
        final_index = index
        #=
        if index == 0 && !(absolute) && int_size_count == 0
            data_int_size[index] = 0
        end
        =#
        
    end
    for i in 1:step:(mini - global_min)
        data_tie_strength[final_index + i] = data_tie_strength[final_index]
        data_num[final_index + i] = data_num[final_index]
        data_degree[final_index + i] = data_degree[final_index]
        #data_int_size[index] = data_int_size[prev_index]
    end
    return data_tie_strength, data_num, data_size, data_degree, data_int_size, count, index
end

function set_data_absolute_time_no_user(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, 
    ego::Int64, global_min::Int64, step::Int64,
    count_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},
    global_max_time::Int64)

    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    end
    
    num_edges = 0
    unique = []
    count = 0
    index = 1
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false
    count_until_ego = 0
    count_without_ego = 0
    ego_found = false
    #get mini value-TODO
    #=
    for (time, simplex) in data
        if simplex == []
            continue
        else
            mini = time
            break
        end
    end=#

    #iterate through dataset
    for (time, simplex_group) in data
        #println("Time: $time, Simplex Group: $simplex_group")
        if simplex_group == []
            continue
        end

        for simplex in simplex_group # changed from shuffle!(simplex_group) because if you shuffle here and in user then order will be different
            #println("No user: time = $time; original simplex = $simplex")
            start = true
            if (ego in simplex)
                ego_found = true
            end

            if !(ego_found)
                count_until_ego += 1
            end

            if !(ego in simplex)
                count_without_ego += 1
            end

            #println("Simplex before = $simplex")
            simplex = filter(e->e!=ego,simplex)
            
            # This shouldn't be doing anything right?
            if simplex == []
                println("Simplex empty")
                #println("Time: $time, Simplex: $simplex")
                continue
            end

            length_simplex = length(simplex)
            final_time = time
            push!(simplex_history, simplex)

            #calculate degrees per node for each iteration        
            for node in simplex
                if node in keys(degrees)
                    degrees[node] += 1
                else
                    degrees[node] = 1
                end
            end

            count += 1

            new_nodes = [node for node in simplex if !(node in unique)]
            #println("New nodes: $new_nodes, Unique: $unique")
            Threads.@threads for i = 1:2
                if i == 1
                    combs = combinations(simplex)
                    for x in combs
                        if length(x) != 2
                            continue
                        elseif get(pairs, sort(x), 0) == 0
                            pairs[sort(x)] = 1
                        else
                            pairs[sort(x)] += 1
                        end
                    end
                elseif i == 2
                    for x in simplex
                        if !(x in unique)
                            push!(unique, x)
                        end
                    end
                end
            end

            if (haskey(count_dict, index))
                count_dict[index] += 1
            else
                count_dict[index] = 1
            end

            length_unique = length(unique)
            num_edges = length(keys(pairs))
            prev_index = index
            #println("No user: time = $time; simplex = $simplex; number of edges = $num_edges; unique = $unique; number of nodes = $(length(unique))")
            
            if (num_edges == 0)
                data_tie_strength[time] = Float64(0)
                data_size[time] = length_simplex
                data_num[time] = length_unique
                data_degree[time] = Float64(1) #sum(values(degrees))/ length_unique
                data_novelty[time] = length(new_nodes)
                index += 1
                continue
            end

            data_tie_strength[time] = Float64(sum(values(pairs)) / num_edges)
            data_num[time] = length_unique
            data_size[time] = length_simplex
            data_degree[time] = sum(values(degrees))/ length_unique
            data_novelty[time] = length(new_nodes)
            final_index = index
            index += 1
        end
        #println()
    end
    #=
    println("data_ratio")
    for (k, v) in data_ratio
        println("Time = $k, num = $v")
    end
    println()
    =#
    #global_max_time = 8000

    # Get avg growth rate
    ratios_to_avg = []
    avg_growth_rate = 0
    for (k, v) in data_novelty
        push!(ratios_to_avg, v)
    end
    popfirst!(ratios_to_avg)
    if !(isempty(ratios_to_avg))
        avg_growth_rate = mean(ratios_to_avg)
    end

    sizes_to_avg = []
    for (k, v) in data_size
        push!(sizes_to_avg, v)
    end
    avg_size = mean(sizes_to_avg)
    #=
    if final_index != 0
        if (final_index < global_max_time)
            for i in final_index:global_max_time
                if (haskey(count_dict, i))
                    count_dict[i] += 1
                else
                    count_dict[i] = 1
                end
                data_novelty[i] = data_novelty[final_index]
                data_tie_strength[i] = data_tie_strength[final_index]
                data_num[i] = data_num[final_index]
                data_degree[i] = data_degree[final_index]
            end
        end
    end
    =#
    #println("until: $count_until_ego")
    #println("without: $count_without_ego")
    frac_simplices_before_ego = count_until_ego / count_without_ego
    #println("frac: $frac_simplices_before_ego")

    return data_tie_strength, data_num, data_size, data_degree, data_novelty, count, final_index, avg_growth_rate, avg_size, frac_simplices_before_ego, count_until_ego
end

function set_data_ordinal_time(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, global_min::Int64, step::Int64,
    count_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, global_max_time::Int64)
    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_ratio = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_ratio = DataStructures.SortedDict()
    end
    
    n = 0
    unique = []
    count = 0
    index = 0
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false
    index = 1

    #iterate through dataset
    for (time, simplex_group) in data # change to for (time, simplex_group) in date
        #println("User: time = $time; simplices = $simplex_group")
        if simplex_group == []
            continue
        end

        for simplex in simplex_group

            if simplex == []
                continue
            end

            length_simplex = length(simplex)
            final_time = time
            push!(simplex_history, simplex)

            #calculate degrees per node for each iteration        
            for node in simplex
                if node in keys(degrees)
                    degrees[node] += 1
                else
                    degrees[node] = 1
                end
            end

            count += 1

            new_nodes = [node for node in simplex if !(node in unique)]

            Threads.@threads for i = 1:2
                if i == 1
                    combs = combinations(simplex)
                    for x in combs
                        if length(x) != 2
                            continue
                        elseif get(pairs, sort(x), 0) == 0
                            pairs[sort(x)] = 1
                        else
                            pairs[sort(x)] += 1
                        end
                    end
                elseif i == 2
                    for x in simplex
                        if !(x in unique)
                            push!(unique, x)
                        end
                    end
                end
            end

            if (haskey(count_dict, index))
                count_dict[index] += 1
            else
                count_dict[index] = 1
            end

            length_unique = length(unique)
            num_edges = length(keys(pairs))
            prev_index = index
            #println("User: time = $time; simplex = $simplex; number of edges = $num_edges; number of nodes = $(length(unique))")
            #=
            if (num_edges == 0)
                data_tie_strength[index] = Float64(0)
                data_size[index] = length_simplex
                data_num[index] = length_unique
                data_degree[index] = Float64(1) # sum(values(degrees))/ length_unique
                #data_int_size[index] = Float64(int_size_sum / int_size_count)
                data_ratio[index] = Float64(0)
                index += 1
                continue
            end
            =#
            data_tie_strength[index] = Float64(sum(values(pairs)) / num_edges)
            data_num[index] = length_unique
            data_size[index] = length_simplex
            data_degree[index] = sum(values(degrees))/ length_unique
            data_ratio[index] = length(new_nodes) / length(simplex)
            #data_int_size[index] = Float64(int_size_sum / int_size_count)

            final_index = index
            index += 1
        end

    end
    
    #println()
    #println("data_n")
    #=
    for (k, v) in data_tie_strength
        if v != Float64(1.0)
            println("Time = $k, num = $v")
        end
        break
    end
    =#
    #println("final_index = $final_index")
    #println("global_max_time = $global_max_time")
    if final_index != 0
        if (final_index < global_max_time)
            #println("TRUEEEEEEEEEEEEEEEEEEEEEEEEEEEE234567890876543213456789")
            for i in final_index:global_max_time
                if (haskey(count_dict, i))
                    count_dict[i] += 1
                else
                    count_dict[i] = 1
                end
                data_ratio[i] = data_ratio[final_index]
                data_tie_strength[i] = data_tie_strength[final_index]
                data_num[i] = data_num[final_index]
                data_degree[i] = data_degree[final_index]
            end
        end
    end
    return data_tie_strength, data_num, data_size, data_degree, data_ratio, count, final_index
end

function set_data_ordinal_time_no_user(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, 
    ego::Int64, global_min::Int64, step::Int64,
    count_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},
    global_max_time::Int64)

    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    end
    
    num_edges = 0
    unique = []
    count = 0
    index = 1
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false
    count_until_ego = 0
    count_without_ego = 0
    ego_found = false
    #get mini value-TODO
    #=
    for (time, simplex) in data
        if simplex == []
            continue
        else
            mini = time
            break
        end
    end=#

    #iterate through dataset
    for (time, simplex_group) in data
        #println("Time: $time, Simplex Group: $simplex_group")
        if simplex_group == []
            continue
        end

        for simplex in simplex_group # changed from shuffle!(simplex_group) because if you shuffle here and in user then order will be different
            #println("No user: time = $time; original simplex = $simplex")
            start = true
            if (ego in simplex)
                ego_found = true
            end

            if !(ego_found)
                count_until_ego += 1
            end

            if !(ego in simplex)
                count_without_ego += 1
            end

            #println("Simplex before = $simplex")
            simplex = filter(e->e!=ego,simplex)
            
            # This shouldn't be doing anything right?
            if simplex == []
                println("Simplex empty")
                #println("Time: $time, Simplex: $simplex")
                continue
            end

            length_simplex = length(simplex)
            final_time = time
            push!(simplex_history, simplex)

            #calculate degrees per node for each iteration        
            for node in simplex
                if node in keys(degrees)
                    degrees[node] += 1
                else
                    degrees[node] = 1
                end
            end

            count += 1

            new_nodes = [node for node in simplex if !(node in unique)]
            #println("New nodes: $new_nodes, Unique: $unique")
            Threads.@threads for i = 1:2
                if i == 1
                    combs = combinations(simplex)
                    for x in combs
                        if length(x) != 2
                            continue
                        elseif get(pairs, sort(x), 0) == 0
                            pairs[sort(x)] = 1
                        else
                            pairs[sort(x)] += 1
                        end
                    end
                elseif i == 2
                    for x in simplex
                        if !(x in unique)
                            push!(unique, x)
                        end
                    end
                end
            end

            if (haskey(count_dict, index))
                count_dict[index] += 1
            else
                count_dict[index] = 1
            end

            length_unique = length(unique)
            num_edges = length(keys(pairs))
            prev_index = index
            #println("No user: time = $time; simplex = $simplex; number of edges = $num_edges; unique = $unique; number of nodes = $(length(unique))")
            
            if (num_edges == 0)
                data_tie_strength[index] = Float64(0)
                data_size[index] = length_simplex
                data_num[index] = length_unique
                data_degree[index] = Float64(1) #sum(values(degrees))/ length_unique
                #data_int_size[index] = Float64(int_size_sum / int_size_count)
                data_novelty[index] = length(new_nodes)
                index += 1
                continue
            end

            data_tie_strength[index] = Float64(sum(values(pairs)) / num_edges)
            data_num[index] = length_unique
            data_size[index] = length_simplex
            data_degree[index] = sum(values(degrees))/ length_unique
            data_novelty[index] = length(new_nodes)
            final_index = index
            index += 1
        end
        #println()
    end
    #=
    println("data_ratio")
    for (k, v) in data_ratio
        println("Time = $k, num = $v")
    end
    println()
    =#
    #global_max_time = 8000

    # Get avg growth rate
    ratios_to_avg = []
    avg_growth_rate = 0
    for (k, v) in data_novelty
        push!(ratios_to_avg, v)
    end
    popfirst!(ratios_to_avg)
    if !(isempty(ratios_to_avg))
        avg_growth_rate = mean(ratios_to_avg)
    end

    sizes_to_avg = []
    for (k, v) in data_size
        push!(sizes_to_avg, v)
    end
    avg_size = mean(sizes_to_avg)
    #=
    if final_index != 0
        if (final_index < global_max_time)
            for i in final_index:global_max_time
                if (haskey(count_dict, i))
                    count_dict[i] += 1
                else
                    count_dict[i] = 1
                end
                data_novelty[i] = data_novelty[final_index]
                data_tie_strength[i] = data_tie_strength[final_index]
                data_num[i] = data_num[final_index]
                data_degree[i] = data_degree[final_index]
            end
        end
    end
    =#
    #println("until: $count_until_ego")
    #println("without: $count_without_ego")
    frac_simplices_before_ego = count_until_ego / count_without_ego
    #println("frac: $frac_simplices_before_ego")

    return data_tie_strength, data_num, data_size, data_degree, data_novelty, count, final_index, avg_growth_rate, avg_size, frac_simplices_before_ego, count_until_ego
end

function set_data_ordinal_time_no_user_set_length(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, 
    ego::Int64, global_min::Int64, step::Int64,
    count_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},
    global_max_time::Int64)

    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    end
    
    num_edges = 0
    unique = []
    count = 0
    index = 1
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false
    count_until_ego = 0
    count_without_ego = 0
    ego_found = false
    count_reached = false

    #get mini value-TODO
    #=
    for (time, simplex) in data
        if simplex == []
            continue
        else
            mini = time
            break
        end
    end=#

    #iterate through dataset
    for (time, simplex_group) in data
        #println("Time: $time, Simplex Group: $simplex_group")
        if simplex_group == []
            continue
        end

        if count_reached
            break
        end

        for simplex in simplex_group # changed from shuffle!(simplex_group) because if you shuffle here and in user then order will be different
            #println("No user: time = $time; original simplex = $simplex")
            start = true
            if (ego in simplex)
                ego_found = true
            end

            if !(ego_found)
                count_until_ego += 1
            end

            if !(ego in simplex)
                count_without_ego += 1
            end

            if (count == global_max_time)
                count_reached = true
                break
            end

            count += 1

            #println("Simplex before = $simplex")
            simplex = filter(e->e!=ego,simplex)
            
            # This shouldn't be doing anything right?
            if simplex == []
                println("Simplex empty")
                #println("Time: $time, Simplex: $simplex")
                continue
            end

            length_simplex = length(simplex)
            final_time = time
            push!(simplex_history, simplex)

            #calculate degrees per node for each iteration        
            for node in simplex
                if node in keys(degrees)
                    degrees[node] += 1
                else
                    degrees[node] = 1
                end
            end

            new_nodes = [node for node in simplex if !(node in unique)]
            #println("New nodes: $new_nodes, Unique: $unique")
            Threads.@threads for i = 1:2
                if i == 1
                    combs = combinations(simplex)
                    for x in combs
                        if length(x) != 2
                            continue
                        elseif get(pairs, sort(x), 0) == 0
                            pairs[sort(x)] = 1
                        else
                            pairs[sort(x)] += 1
                        end
                    end
                elseif i == 2
                    for x in simplex
                        if !(x in unique)
                            push!(unique, x)
                        end
                    end
                end
            end

            if (haskey(count_dict, index))
                count_dict[index] += 1
            else
                count_dict[index] = 1
            end

            length_unique = length(unique)
            num_edges = length(keys(pairs))
            prev_index = index
            #println("No user: time = $time; simplex = $simplex; number of edges = $num_edges; unique = $unique; number of nodes = $(length(unique))")
            
            if (num_edges == 0)
                data_tie_strength[index] = Float64(0)
                data_size[index] = length_simplex
                data_num[index] = length_unique
                data_degree[index] = Float64(1) #sum(values(degrees))/ length_unique
                #data_int_size[index] = Float64(int_size_sum / int_size_count)
                data_novelty[index] = length(new_nodes)
                index += 1
                continue
            end

            data_tie_strength[index] = Float64(sum(values(pairs)) / num_edges)
            data_num[index] = length_unique
            data_size[index] = length_simplex
            data_degree[index] = sum(values(degrees))/ length_unique
            data_novelty[index] = length(new_nodes)
            final_index = index
            index += 1
        end
        #println()
    end
    #=
    println("data_ratio")
    for (k, v) in data_ratio
        println("Time = $k, num = $v")
    end
    println()
    =#
    #global_max_time = 8000

    # Get avg growth rate
    ratios_to_avg = []
    avg_growth_rate = 0
    for (k, v) in data_novelty
        push!(ratios_to_avg, v)
    end
    popfirst!(ratios_to_avg)
    if !(isempty(ratios_to_avg))
        avg_growth_rate = mean(ratios_to_avg)
    end

    sizes_to_avg = []
    for (k, v) in data_size
        push!(sizes_to_avg, v)
    end
    avg_size = mean(sizes_to_avg)
    #=
    if final_index != 0
        if (final_index < global_max_time)
            for i in final_index:global_max_time
                if (haskey(count_dict, i))
                    count_dict[i] += 1
                else
                    count_dict[i] = 1
                end
                data_novelty[i] = data_novelty[final_index]
                data_tie_strength[i] = data_tie_strength[final_index]
                data_num[i] = data_num[final_index]
                data_degree[i] = data_degree[final_index]
            end
        end
    end
    =#
    #println("until: $count_until_ego")
    #println("without: $count_without_ego")
    frac_simplices_before_ego = count_until_ego / count_without_ego
    #println("frac: $frac_simplices_before_ego")

    return data_tie_strength, data_num, data_size, data_degree, data_novelty, count, final_index, avg_growth_rate, avg_size, frac_simplices_before_ego, count_until_ego
end

function set_data_ordinal_time_until_user(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering}, 
    ego::Int64, global_min::Int64, step::Int64,
    count_dict::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering},
    global_max_time::Int64)

    pairs = Dict()
    absolute = false
    if (absolute)
        #data_tie_strength = DataStructures.deepcopy(base)
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    else
        data_tie_strength = DataStructures.SortedDict()
        data_size = DataStructures.SortedDict()
        data_num = DataStructures.SortedDict()
        data_degree = DataStructures.SortedDict()
        data_novelty = DataStructures.SortedDict()
    end
    
    num_edges = 0
    unique = []
    count = 0
    index = 1
    sum_degree_simplex = 0
    degrees = DataStructures.SortedDict()
    simplex_history = []
    int_size_sum = 0
    int_size_count = 0
    mini = 0
    prev_index = 0
    final_time = 0
    final_index = 0
    start = false
    count_until_ego = 0
    count_without_ego = 0
    ego_found = false

    #iterate through dataset
    for (time, simplex_group) in data
        if simplex_group == []
            continue
        end

        for simplex in simplex_group
            start = true

            if (ego in simplex)

                # Handle avg degrees
                degs_to_avg = []
                simplex = filter(e->e!=ego,simplex)
                for node in simplex
                    if node in keys(degrees)
                        push!(degs_to_avg, degrees[node])
                    else
                        push!(degs_to_avg, 0)
                    end
                end
                avg_deg_simplex_with_ego = mean(degs_to_avg)
                if isempty(degrees)
                    avg_degree = 0
                else
                    avg_degree = mean(values(degrees))
                end
                
                # Handle avg simplex size
                avg_size = 0
                sizes_to_avg = []
                for (k, v) in data_size
                    push!(sizes_to_avg, v)
                end
                if !(isempty(data_size))
                    avg_size = mean(sizes_to_avg)
                end
                simplex_with_ego_size = length(simplex)
                
                # Handle novelty rate
                new_nodes = [node for node in simplex if !(node in unique)]
                simplex_with_ego_novelty = length(new_nodes)
                avg_novelty = 0
                novelty_to_average = []
                for (k, v) in data_novelty
                    push!(novelty_to_average, v)
                end
                if !(isempty(data_novelty))
                    avg_novelty = mean(novelty_to_average)
                end

                return avg_degree, avg_deg_simplex_with_ego, avg_size, simplex_with_ego_size, avg_novelty, simplex_with_ego_novelty, index
            end

            if !(ego_found)
                count_until_ego += 1
            end

            if !(ego in simplex)
                count_without_ego += 1
            end

            #println("Simplex before = $simplex")
            simplex = filter(e->e!=ego,simplex)
            
            # This shouldn't be doing anything right?
            if simplex == []
                println("Simplex empty")
                continue
            end

            length_simplex = length(simplex)
            final_time = time
            push!(simplex_history, simplex)

            #calculate degrees per node for each iteration        
            for node in simplex
                if node in keys(degrees)
                    degrees[node] += 1
                else
                    degrees[node] = 1
                end
            end

            count += 1

            new_nodes = [node for node in simplex if !(node in unique)]
            #println("New nodes: $new_nodes, Unique: $unique")
            Threads.@threads for i = 1:2
                if i == 1
                    combs = combinations(simplex)
                    for x in combs
                        if length(x) != 2
                            continue
                        elseif get(pairs, sort(x), 0) == 0
                            pairs[sort(x)] = 1
                        else
                            pairs[sort(x)] += 1
                        end
                    end
                elseif i == 2
                    for x in simplex
                        if !(x in unique)
                            push!(unique, x)
                        end
                    end
                end
            end

            if (haskey(count_dict, index))
                count_dict[index] += 1
            else
                count_dict[index] = 1
            end

            length_unique = length(unique)
            num_edges = length(keys(pairs))
            prev_index = index
            #println("No user: time = $time; simplex = $simplex; number of edges = $num_edges; unique = $unique; number of nodes = $(length(unique))")
            
            if (num_edges == 0)
                data_tie_strength[index] = Float64(0)
                data_size[index] = length_simplex
                data_num[index] = length_unique
                data_degree[index] = Float64(1) #sum(values(degrees))/ length_unique
                #data_int_size[index] = Float64(int_size_sum / int_size_count)
                data_novelty[index] = length(new_nodes)
                index += 1
                continue
            end

            data_tie_strength[index] = Float64(sum(values(pairs)) / num_edges)
            data_num[index] = length_unique
            data_size[index] = length_simplex
            data_degree[index] = sum(values(degrees))/ length_unique
            data_novelty[index] = length(new_nodes)
            final_index = index
            index += 1
        end
        #println()
    end
    #=
    println("data_ratio")
    for (k, v) in data_ratio
        println("Time = $k, num = $v")
    end
    println()
    =#
    #global_max_time = 8000

    # Get avg growth rate
    ratios_to_avg = []
    for (k, v) in data_novelty
        push!(ratios_to_avg, v)
    end
    popfirst!(ratios_to_avg)
    avg_growth_rate = mean(ratios_to_avg)

    sizes_to_avg = []
    for (k, v) in data_size
        push!(sizes_to_avg, v)
    end
    avg_size = mean(sizes_to_avg)
    #=
    if final_index != 0
        if (final_index < global_max_time)
            for i in final_index:global_max_time
                if (haskey(count_dict, i))
                    count_dict[i] += 1
                else
                    count_dict[i] = 1
                end
                data_novelty[i] = data_novelty[final_index]
                data_tie_strength[i] = data_tie_strength[final_index]
                data_num[i] = data_num[final_index]
                data_degree[i] = data_degree[final_index]
            end
        end
    end
    =#
    #println("until: $count_until_ego")
    #println("without: $count_without_ego")
    frac_simplices_before_ego = count_until_ego / count_without_ego
    #println("frac: $frac_simplices_before_ego")

    return data_tie_strength, data_num, data_size, data_degree, data_novelty, count, index, avg_growth_rate, avg_size, frac_simplices_before_ego, count_until_ego
end

function set_data_ordinal_time_duplicates(data::DataStructures.SortedDict{Any,Any,Base.Order.ForwardOrdering})

    duplicates = []
    unique = []
    #iterate through dataset
    for (time, simplex_group) in data
        if simplex_group == []
            continue
        end

        for simplex in simplex_group
            if simplex in unique
                push!(duplicates, simplex)
            else
                push!(unique, simplex)
            end
        end        
    end

    duplicates_dict = Dict()
    # If a simplex does have a duplicate, what’s the probability that the duplicate is in an adjacent spot? Within 2 spots? 3 spots?
    # keys in dict should be 0 or 1, and then mean(keys(dict))
    for duplicate_simplex in duplicates
        start_count = false
        count = 0
        counts = []
        duplicate_time = 0
        for (time, simplex_group) in data
            if simplex_group == []
                continue
            end

            for simplex in simplex_group
                if (simplex == duplicate_simplex) && !(start_count)
                    start_count = true
                    duplicate_time = time
                    println("Duplicate Simplex: $duplicate_simplex, Duplicate Time: $duplicate_time")
                elseif (simplex == duplicate_simplex) && start_count
                    #count += 1
                    println("Simplex: $simplex, Time: $time, Time - duplicate_time = $(time - duplicate_time)")
                    if time - duplicate_time == 0
                        duplicates_dict[duplicate_simplex] = 1
                    else
                        duplicates_dict[duplicate_simplex] = 0
                    end
                    println()
                    @goto next
                end
                if start_count
                    #count += 1
                end
            end        
        end
        @label next
    end

    return duplicates_dict
end

# This is just a convenient wrapper around all of the formatting parameters for
# making plots.
function all_datasets_params()
    green  = "#1b9e77"
    orange = "#d95f02"
    purple = "#7570b3"
    plot_params = [["coauth-DBLP",            "x", green],
                   ["coauth-MAG-Geology",     "x", orange],
                   ["coauth-MAG-History",     "x", purple],
                   ["music-rap-genius",       "v", green],
                   ["tags-stack-overflow",    "s", green],
                   ["tags-math-sx",           "s", orange],
                   ["tags-ask-ubuntu",        "s", purple],
                   ["threads-stack-overflow", "o", green],
                   ["threads-math-sx",        "o", orange],
                   ["threads-ask-ubuntu",     "o", purple],
                   ["NDC-substances",         "<", green],
                   ["NDC-classes",            "<", orange],
                   ["DAWN",                   "p", green],
                   ["congress-bills",         "*", green],
                   ["congress-committees",    "*", orange],
                   ["email-Eu",               "P", green],
                   ["email-Enron",            "P", orange],
                   ["contact-high-school",    "d", green],
                   ["contact-primary-school", "d", orange],
                   ]
    return plot_params
end
;
