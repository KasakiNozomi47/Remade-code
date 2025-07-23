##########################
#Part 1: 我们在此先生成了每个点的运动历史，即序列ηk，其中共4097个点，6000步历史，先固定alpha，随后再通过改变alpha的值检测其方差与
using StatsBase
using HypothesisTests
using Distributions

x_min = -2048                             #2048
x_max = 2048                              #2048
theta = 0                               # 质量幂次参数，我们依次取0，1/2，1，2，10，无穷
num_steps = 2000                          #分别考虑100，200……2000步，步差为100，
num_steps_history = 6000                  #ηk的长度
num_points = x_max - x_min + 1            #点的个数
#point_history = []                       #ηk的记录
alpha =  1/6                              #α的取值
H = alpha + 1/2                           #Hurst Coefficient

#我们先一次性取出10^8个随机数
#A = rand(10^8)

# 优化：预分配Yita和History，避免push!
Yita = Vector{Vector{Float64}}(undef, num_points)
History = Vector{Vector{Int}}(undef, num_points)

# 定义函数g
function g(x)
    return (1 - x)^(-1 / alpha) - 1
end

# 生成单个点的历史数据的函数
function generate_point_history(num_steps_history, alpha)
    """
    为单个点生成历史数据
    
    参数:
    - num_steps_history: 历史长度
    - alpha: α参数
    
    返回:
    - (random_numbers, history): 随机数序列和对应的历史序列
    """
    # 生成随机数序列
    random_numbers = rand(num_steps_history)
    
    # 初始化历史序列
    history_each_point = Vector{Int}(undef, num_steps_history)
    
    # 定义本地g函数（避免全局变量依赖）
    g_local(x) = (1 - x)^(-1 / alpha) - 1
    
    for j in 1:num_steps_history
        x = random_numbers[j]
        result = g_local(x)
        result = ceil(result)
        
        if result > num_steps_history
            index = -1
        else
            result = Int(result)
            index = j - result
        end
        
        if index <= 0
            random_walk = rand([-1, 1])
        else
            random_walk = history_each_point[index]
        end
        
        history_each_point[j] = random_walk
    end
    
    return random_numbers, history_each_point
end

# 生成所有点的历史数据的函数
function generate_all_histories(num_points, num_steps_history, alpha)
    """
    为所有点生成历史数据
    
    参数:
    - num_points: 点的数量
    - num_steps_history: 历史长度
    - alpha: α参数
    
    返回:
    - (Yita, History): 随机数矩阵和历史矩阵
    """
    println("=== 开始生成历史数据 ===")
    history_start_time = time()
    
    # 预分配数组
    Yita = Vector{Vector{Float64}}(undef, num_points)
    History = Vector{Vector{Int}}(undef, num_points)
    
    # 使用多线程并行生成
    Threads.@threads for i in 1:num_points
        random_numbers, history = generate_point_history(num_steps_history, alpha)
        Yita[i] = random_numbers
        History[i] = history
    end
    
    history_end_time = time()
    history_elapsed_time = history_end_time - history_start_time
    println("=== 历史数据生成完成 ===")
    println("历史数据生成时间: $(round(history_elapsed_time, digits=2)) 秒")
    println("线程数: $(Threads.nthreads())")
    println()
    
    return Yita, History
end

# 调用函数生成历史数据
Yita, History = generate_all_histories(num_points, num_steps_history, alpha)

#综上我们成功定义了所需要的ηk序列并且将其保存在了History数组中以便调用
#下面我们将考虑如何对点的随机游走进行处理



################################################################################### 以上完成了历史数据的生成
#Part 2 ：对于每个点，先进行一定规则的随机游走，再判断点的运动状态以及吸收态，
B = rand(10^8)  # 生成一个包含10^8个随机数的数组
# 定义结构体
struct Point
    coords::Vector{Int}   # 点的坐标
    i::Int                # 点的索引，对点就行标号，判断点的吸收态
    j::Int                # 点被哪个点所吸收,点的吸收态
    k::Int                # 点的质量
    t::Int                # 0 为可以移动，1 为不能移动
    history::Vector{Int}  # 点的历史，存为一个数组
end

# 初始化所有点
function init_points(x_min, x_max, motion_history)
    points = [Point([2i, 0], i, i, 1, 0, motion_history[i - x_min + 1]) for i in x_min:x_max]
    return points
end

# 初始化轨迹
function init_trajectories(points, x_min)
    trajectories = [Vector{Vector{Int}}() for _ in points]
    for (index, point) in enumerate(points)
        idx = point.i - x_min + 1
        push!(trajectories[idx], point.coords)
    end
    return trajectories
end

# 构建可移动点的坐标字典
function build_coord_dict(points)
    coord_dict = Dict{Tuple{Int,Int}, Int}()
    for (idx, p) in enumerate(points)
        if p.t == 0
            coord_dict[(p.coords[1], p.coords[2])] = idx
        end
    end
    return coord_dict
end

# 生成本步所有可动点的随机数
function get_random_numbers(B, num_movable)
    random_numbers = splice!(B, 1:num_movable)
    return random_numbers
end

# 计算本步所有可动点的结果
function get_results(random_numbers, alpha)
    results = map(η -> (1 - η) ^ (-1 / alpha) - 1, random_numbers)
    results = ceil.(results)
    return results
end

# 添加级联更新j值的辅助函数
function cascade_update_j!(points, old_j, new_j)
    for (idx, p) in enumerate(points)
        if p.j == old_j
            points[idx] = Point(p.coords, p.i, new_j, p.k, p.t, p.history)
        end
    end
end


# 单步模拟
function simulate_step!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min)
    coord_dict = build_coord_dict(points)
    movable_points = filter(p -> p.t == 0, points)
    num_movable = length(movable_points)
    random_numbers = get_random_numbers(B, num_movable)
    results = get_results(random_numbers, alpha)

    for (index, point) in enumerate(movable_points)
        coord_dict = build_coord_dict(points)
        global_idx = findfirst(p -> p.i == point.i, points)
        if points[global_idx].t == 1
            continue
        end
        if point.t == 0
            movable_index = findfirst(p -> p.i == point.i, movable_points)
            if  movable_index !== nothing
                result = results[movable_index]
                if  result > num_steps + num_steps_history
                    random_change = rand([-1, 1])
                else
                    result = Int(result)
                    if  length(point.history) - result <= 0
                        random_change = rand([-1, 1])
                    else
                        random_change = point.history[length(point.history) - result]
                    end
                end
            end

            new_history = copy(point.history)
            push!(new_history, random_change)
            new_coords = copy(point.coords)
            new_coords[1] += random_change
            new_coords[2] += 1
            
            new_point = Point(new_coords, point.i, point.j, point.k, 0, new_history)
            global_idx = findfirst(p -> p.i == point.i, points)
            points[global_idx] = new_point

            new_coord_tuple = (new_coords[1], new_coords[2])
            merged = false
            if  haskey(coord_dict, new_coord_tuple)
                target_idx = coord_dict[new_coord_tuple]
                if points[target_idx].i != point.i && points[target_idx].t == 0
                    m1 = new_point.k
                    m2 = points[target_idx].k
                    prob = m1^theta / (m1^theta + m2^theta)
                    
                    if  rand() < prob 
                        # global_idx wins - 级联更新所有被target吸收的点
                        old_j_target = points[target_idx].j
                        new_j_winner = points[global_idx].j
                        
                        # 更新获胜者的质量
                        points[global_idx] = Point(points[global_idx].coords, points[global_idx].i, points[global_idx].j, m1 + m2, 0, points[global_idx].history)
                        
                        # 级联更新所有j值等于old_j_target的点
                        cascade_update_j!(points, old_j_target, new_j_winner)
                        
                        # 设置被合并点为不可动
                        points[target_idx] = Point(points[target_idx].coords, points[target_idx].i, new_j_winner, points[target_idx].k, 1, points[target_idx].history)
                        merged = true
                    else                
                        # target_idx wins - 级联更新所有被global_idx吸收的点
                        old_j_global = points[global_idx].j
                        new_j_winner = points[target_idx].j
                        
                        # 更新获胜者的质量
                        points[target_idx] = Point(points[target_idx].coords, points[target_idx].i, points[target_idx].j, m1 + m2, 0, points[target_idx].history)
                        
                        # 级联更新所有j值等于old_j_global的点
                        cascade_update_j!(points, old_j_global, new_j_winner)
                        
                        # 设置被合并点为不可动
                        points[global_idx] = Point(points[global_idx].coords, points[global_idx].i, new_j_winner, points[global_idx].k, 1, points[global_idx].history)
                        merged = true
                    end
                    
                    # 合并后立即更新coord_dict，确保t值最新
                    coord_dict = build_coord_dict(points)
                    # 合并后也push最后一次坐标
                    idx = point.i - x_min + 1
                    push!(trajectories[idx], new_point.coords)
                    continue
                end
            end
            # 只要点还没被合并（t=0），就持续push
            if points[global_idx].t == 0 && !merged
                idx = point.i - x_min + 1
                push!(trajectories[idx], new_point.coords)
            end
        end
    end
end


#Part 3: 数据操作和统计分析
using Test
#以上完成了模拟，
#下面进行一定的数据操作
function analyze_step(trajectories, points, step, x_min)
    #for step in 100:100:num_steps
        # 统计每个点在step步时的横坐标和初始横坐标
        coord_map = Dict{Int, Tuple{Int, Int, Int}}()  # key: x坐标, value: (idx, t, origin)
        for (idx, traj) in enumerate(trajectories)
            if length(traj) >= step
                x = traj[step][1]
                origin = traj[1][1]
                tval = points[idx].t
                if haskey(coord_map, x)
                    # 若有重叠，保留t=0的，删除t=1的
                    old_idx, old_t, old_origin = coord_map[x]
                    if old_t == 1 && tval == 0
                    coord_map[x] = (idx, tval, origin)
                    elseif old_t == 0 && tval == 1
                        # reserve the original
                    elseif old_t == 0 && tval == 0
                        # 都可动，任选其一（默认保留原有）
                    elseif old_t == 1 && tval == 1
                        # 都不可动，任选其一（默认保留原有）
                    end
                else
                    coord_map[x] = (idx, tval, origin)
                end
            end
        end
    
    # 按照x_coords从小到大排序
    sorted_keys = sort(collect(keys(coord_map)))
    x_coords = sorted_keys
    origins = [coord_map[k][3] for k in sorted_keys]
    @assert length(x_coords) == length(origins) "Length mismatch: coords_diff and origins should be the same length"
    println("Step $step: x_coords = $x_coords")
    println("Step $step: origins = $origins")

    # 计算每个点的位移
    Displacement = [x_coords[i]- origins[i] for i in 1:length(x_coords)]
    println("Step $step: Displacement = $Displacement")
    @assert length(Displacement) == length(origins) "Length mismatch: Displacement and origins should be the same length"

    # 检验每个点的j值
    values_of_j = [points[coord_map[x][1]].j for x in sorted_keys]
    #println("Step $step: masses = $masses")
    @assert length(values_of_j) == length(origins) "Length mismatch: masses and origins should be the same length"

    # 计算吸收态相关的最大i值差距
    coords_diff = []
    for origin_index in origins
        k_idx = findfirst(x -> x.j == origin_index/2, points)
        m_idx = findlast(x -> x.j == origin_index/2, points)

        if k_idx !== nothing && m_idx !== nothing
            coord_diff = abs(points[k_idx].i - points[m_idx].i)
            push!(coords_diff, coord_diff)
        else
            # 如果没找到，设为0或跳过，视需求而定
            push!(coords_diff, 0)  # 或者 continue
        end
    end
    coords_difference = [coords_diff[i] for i in 2:length(coords_diff)-1]  # 考虑到边际效应，把最边上的两段去掉了
    println("Step $step: coords_difference = $coords_difference")
    #println(length(coords_diff))
    # 检查coords_diff和origins的长度是否一致
    #print( length(origins) == length( coords_diff)) 
    #print(length(origins) == length(coords_diff) ? "Lengths match" : "Lengths mismatch")

    # Nowwww we calculate the distance of x_coords
    x_coords_diff = [x_coords[i] - x_coords[i-1] for i in 3:length(x_coords)-1]   #考虑到边际效应，把最边上的两段去掉了
    println("Step $step: x_coords_diff = $x_coords_diff")
    #println(length(x_coords_diff) == length(coords_diff)-1)

end
#以上完成了对点的数据的输出




# 主模拟函数
function run_simulation!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min)
    for step in 1:num_steps
        simulate_step!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min)
        if  step ∈ 100:100:num_steps      # 只在这些步输出
            analyze_step(trajectories, points, step, x_min)
            println()
        end
    end
end

# ========== 主流程调用 ===========
println("=== 模型1开始运行 ===")
start_time = time()
points = init_points(x_min, x_max, History)
trajectories = init_trajectories(points, x_min)
run_simulation!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min)
end_time = time()
elapsed_time = end_time - start_time
println("=== 模型1运行完成 ===")
println("模型1总运行时间: $(round(elapsed_time, digits=2)) 秒")
println("模型1总运行时间: $(round(elapsed_time/60, digits=2)) 分钟")
println()

###Part 4: 可视化轨迹 主要用于检测
using Plots
function plot_trajectories(trajectories)
    p = plot()  # 创建一个新的图形
    # 只遍历前100个点的轨迹
    for i in 1:min(200, length(trajectories))
        traj = trajectories[i]
        if !isempty(traj)
            x_coords = [coord[1] for coord in traj]
            y_coords = [coord[2] for coord in traj]
            plot!(p, x_coords, y_coords, legend=false)
        end
    end
    display(p)  # 显示图形
end

# 假设trajectories是之前模拟生成的轨迹数组
println("=== 开始绘制轨迹图 ===")
plot_start_time = time()
plot_trajectories(trajectories)
plot_end_time = time()
plot_elapsed_time = plot_end_time - plot_start_time
println("=== 轨迹绘制完成 ===")
println("轨迹绘制时间: $(round(plot_elapsed_time, digits=2)) 秒")
println()


###project 2：在t=100,200,300……2000时,加入新的点
# Project 2：在t=100,200,300……2000时,加入新的点

println("=== 模型2开始运行 ===")
model2_start_time = time()

# 第二个模型的主要函数
function add_new_points_at_step!(points, trajectories, step, x_min, x_max)
    """
    在指定步骤加入新点，横坐标和原先一样，纵坐标为当前步数
    """
    new_points_added = 0
    for i in x_min:x_max
        # 检查这个横坐标位置是否已经有点了
        existing_point_idx = findfirst(p -> p.coords[1] == 2*i && p.coords[2] == step, points)
        
        if existing_point_idx === nothing
            # 创建新点，使用已有的History
            new_point = Point([2*i, step], i, i, 1, 0, History[i - x_min + 1])
            push!(points, new_point)
            
            # 为新点创建轨迹，初始坐标就是当前位置
            push!(trajectories, Vector{Vector{Int}}())
            push!(trajectories[end], [2*i, step])
            
            new_points_added += 1
        end
    end
    
    return new_points_added
end


##############################################
##################################################
#####Model 2：在特定步骤加入新点的模拟函数########

# 修改的模拟函数，支持在特定步骤加入新点
function run_simulation_with_new_points!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min, x_max)
    # 定义在哪些步骤加入新点
    add_points_steps = 100:100:num_steps
    
    for step in 1:num_steps
        # 在指定步骤加入新点
        if step in add_points_steps
            new_added = add_new_points_at_step!(points, trajectories, step, x_min, x_max)
            println("Step $step: 加入了 $new_added 个新点")
        end
        
        # 执行常规的模拟步骤
        simulate_step_model2!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min)
        
        # 只在特定步输出分析
        if step ∈ 100:100:num_steps
            analyze_step_model2(trajectories, points, step, x_min)
            println()
        end
    end
end

# 修改的单步模拟函数，适应动态点数量
function simulate_step_model2!(points, trajectories, B, alpha, theta, num_steps, num_steps_history, x_min)
    coord_dict = build_coord_dict(points)
    movable_points = filter(p -> p.t == 0, points)
    num_movable = length(movable_points)
    
    # 确保有足够的随机数
    if length(B) < num_movable
        append!(B, rand(num_movable - length(B)))
    end
    
    random_numbers = splice!(B, 1:num_movable)
    results = get_results(random_numbers, alpha)

    for (index, point) in enumerate(movable_points)
        coord_dict = build_coord_dict(points)
        global_idx = findfirst(p -> p.i == point.i && p.coords == point.coords, points)
        
        if global_idx === nothing || points[global_idx].t == 1
            continue
        end
        
        if point.t == 0
            movable_index = findfirst(p -> p.i == point.i && p.coords == point.coords, movable_points)
            if movable_index !== nothing
                result = results[movable_index]
                if result > num_steps + num_steps_history
                    random_change = rand([-1, 1])
                else
                    result = Int(result)
                    if length(point.history) - result <= 0
                        random_change = rand([-1, 1])
                    else
                        random_change = point.history[length(point.history) - result]
                    end
                end
            end

            new_history = copy(point.history)
            push!(new_history, random_change)
            new_coords = copy(point.coords)
            new_coords[1] += random_change
            new_coords[2] += 1
            
            new_point = Point(new_coords, point.i, point.j, point.k, 0, new_history)
            points[global_idx] = new_point

            # 找到对应的轨迹索引并更新
            traj_idx = findfirst(t -> !isempty(t) && t[1] == point.coords, trajectories)
            if traj_idx !== nothing && points[global_idx].t == 0
                push!(trajectories[traj_idx], new_coords)
            end
            
            # ...existing collision and merging logic...
            new_coord_tuple = (new_coords[1], new_coords[2])
            merged = false
            if haskey(coord_dict, new_coord_tuple)
                target_idx = coord_dict[new_coord_tuple]
                if points[target_idx].i != point.i && points[target_idx].t == 0
                    m1 = new_point.k
                    m2 = points[target_idx].k
                    prob = m1^theta / (m1^theta + m2^theta)
                    
                    if  rand() < prob 
                        # global_idx wins - 级联更新所有被target吸收的点
                        old_j_target = points[target_idx].j
                        new_j_winner = points[global_idx].j
                        
                        # 更新获胜者的质量
                        points[global_idx] = Point(points[global_idx].coords, points[global_idx].i, points[global_idx].j, m1 + m2, 0, points[global_idx].history)
                        
                        # 级联更新所有j值等于old_j_target的点
                        cascade_update_j!(points, old_j_target, new_j_winner)
                        
                        # 设置被合并点为不可动
                        points[target_idx] = Point(points[target_idx].coords, points[target_idx].i, new_j_winner, points[target_idx].k, 1, points[target_idx].history)
                        merged = true
                    else                
                        # target_idx wins - 级联更新所有被global_idx吸收的点
                        old_j_global = points[global_idx].j
                        new_j_winner = points[target_idx].j
                        
                        # 更新获胜者的质量
                        points[target_idx] = Point(points[target_idx].coords, points[target_idx].i, points[target_idx].j, m1 + m2, 0, points[target_idx].history)
                        
                        # 级联更新所有j值等于old_j_global的点
                        cascade_update_j!(points, old_j_global, new_j_winner)
                        
                        # 设置被合并点为不可动
                        points[global_idx] = Point(points[global_idx].coords, points[global_idx].i, new_j_winner, points[global_idx].k, 1, points[global_idx].history)
                        merged = true
                    end
                    
                    # 合并后立即更新coord_dict，确保t值最新
                    coord_dict = build_coord_dict(points)
                    # 合并后也push最后一次坐标
                    idx = point.i - x_min + 1
                    push!(trajectories[idx], new_point.coords)
                    continue
                end
            end
            # 只要点还没被合并（t=0），就持续push
            if points[global_idx].t == 0 && !merged
                idx = point.i - x_min + 1
                push!(trajectories[idx], new_point.coords)
            end
        end
    end
end


# 修改的分析函数
function analyze_step_model2(trajectories, points, step, x_min)
    active_points = filter(p -> p.t == 0, points)
    println("Step $step: 当前活跃点数量: $(length(active_points))")
    println("Step $step: 总点数量: $(length(points))")
    
    # 统计不同纵坐标的点的分布
    y_coords = [p.coords[2] for p in active_points]
    unique_y = unique(y_coords)
    
    println("Step $step: 不同纵坐标层的点数分布:")
    for y in sort(unique_y)
        count = sum(coord[2] == y for coord in y_coords)
        println("  y=$y: $count 个点")
    end
end

println("第二个模型函数定义完成！")
model2_end_time = time()
model2_elapsed_time = model2_end_time - model2_start_time
println("=== 模型2定义完成 ===")
println("模型2函数定义时间: $(round(model2_elapsed_time, digits=2)) 秒")
println("使用方法：")
println("1. 重新初始化: points2 = init_points(x_min, x_max, History)")
println("2. 重新初始化轨迹: trajectories2 = init_trajectories(points2, x_min)")
println("3. 运行第二个模型: run_simulation_with_new_points!(points2, trajectories2, rand(10^8), alpha, theta, num_steps, num_steps_history, x_min, x_max)")
println()





###########
#########
############ Part 5: KS Test部分##########

# 统计每个step的Displacement方差，并与t^(2H)做比例性检测
using Statistics
using GLM
using DataFrames  
using HypothesisTests  
using Logging

println("=== 开始KS检验分析 ===")
ks_start_time = time()

# 暂时关闭警告
old_logger = global_logger(SimpleLogger(stderr, Logging.Error))

steps = 100:100:num_steps
vars = Float64[]
ts = Float64[]
all_displacements = Dict{Int, Vector{Float64}}()

for step in steps
    coord_map = Dict{Int, Tuple{Int, Int, Int}}()
    for (idx, traj) in enumerate(trajectories)
        if length(traj) >= step
            x = traj[step][1]
            origin = traj[1][1]
            tval = points[idx].t
            if haskey(coord_map, x)
                old_idx, old_t, old_origin = coord_map[x]
                if old_t == 1 && tval == 0
                    coord_map[x] = (idx, tval, origin)
                end
            else
                coord_map[x] = (idx, tval, origin)
            end
        end
    end
    sorted_keys = sort(collect(keys(coord_map)))
    x_coords = sorted_keys
    origins = [coord_map[k][3] for k in sorted_keys]
    Displacement = [x_coords[i] - origins[i] for i in 1:length(x_coords)]
    
    all_displacements[step] = copy(Displacement)
    
    if length(Displacement) > 1
        push!(vars, var(Displacement))
        push!(ts, step^(2*H))
    end
end

println("H value: ", H)

# 检查方差与t^(2H)的线性相关性
if length(vars) > 1 && length(ts) == length(vars)
    using StatsBase
    r = cor(ts, vars)
    println("Correlation between var(Displacement) and t^(2H): ", r)
    
    # 线性回归
    lm_model = lm(@formula(y ~ x), DataFrame(x=ts, y=vars))
    coef_ = coef(lm_model)
    c_coefficient = round(coef_[2], digits=2)
    intercept = coef_[1]
    
    println("Linear regression results:")
    println("var(Displacement) = ", c_coefficient, " * t^(2H) + ", intercept)
    println("Coefficient c = ", c_coefficient)
    println("R-squared: ", r2(lm_model))
    
    # KS检验
    ks_results = []
    for (i, step) in enumerate(steps)
        if haskey(all_displacements, step) && length(all_displacements[step]) > 1
            theoretical_var = c_coefficient * step^(2*H)
            theoretical_std = sqrt(theoretical_var)
            displacement_data = all_displacements[step]
            normal_dist = Normal(0, theoretical_std)
            ks_test = ApproximateOneSampleKSTest(displacement_data, normal_dist)
            p_value = pvalue(ks_test)
            statistic_value = ks_test.δ
            push!(ks_results, (step=step, p_value=p_value, statistic=statistic_value))
        end
    end
    
    global_logger(old_logger)
    
    # 只输出一次结果
    if length(ks_results) > 0
        p_values = [r.p_value for r in ks_results]
        non_significant_05 = sum(p_values .>= 0.05)
        proportion_ns = non_significant_05 / length(p_values)
        
        println("\n=== 综合结论 ===")
        println("总共进行了 $(length(p_values)) 次KS检验")
        println("在α = 0.05水平下，$(round(proportion_ns*100, digits=1))% 的检验不显著")
                
        ks_df = DataFrame(
            Step = [r.step for r in ks_results],
            P_Value = [round(r.p_value, digits=4) for r in ks_results],
            Statistic = [round(r.statistic, digits=4) for r in ks_results],
            Significant = [r.p_value < 0.05 ? "Yes" : "No" for r in ks_results]
        )
        
        println("\nKS Test Results Table:")
        println(ks_df)
    end
    
    # 绘图
    using Plots
    
    p1 = scatter(ts, vars, xlabel="t^(2H)", ylabel="Var(Displacement)", 
                label="Data Points", legend=:topleft, 
                title="Displacement Variance vs t^(2H), H=$(round(H, digits=3))",
                markersize=5, markercolor=:blue)
    plot!(p1, ts, predict(lm_model), label="Linear Fit (c=$c_coefficient)", 
          linewidth=2, linecolor=:red)
    
    if length(ks_results) > 0
        ks_steps = [r.step for r in ks_results]
        ks_pvalues = [r.p_value for r in ks_results]
        
        p2 = plot(ks_steps, ks_pvalues, 
                 xlabel="Step Number", ylabel="KS Test p-value", 
                 title="KS Test p-values vs Step Number",
                 linewidth=2, marker=:circle, markersize=5,
                 label="KS p-values", legend=:topright,
                 markercolor=:blue, linecolor=:blue)
        
        hline!(p2, [0.05], linestyle=:dash, linecolor=:red, 
               linewidth=2, label="p = 0.05 significance level")
        
        combined_plot = plot(p1, p2, layout=(2,1), size=(800, 700))
        display(combined_plot)
    else
        display(p1)
    end
    
else
    println("数据不足，无法进行相关性分析。")
end

ks_end_time = time()
ks_elapsed_time = ks_end_time - ks_start_time
println("=== KS检验分析完成 ===")
println("KS检验分析时间: $(round(ks_elapsed_time, digits=2)) 秒")
println("KS检验分析时间: $(round(ks_elapsed_time/60, digits=2)) 分钟")

# 计算总运行时间
total_elapsed_time = elapsed_time + model2_elapsed_time + plot_elapsed_time + ks_elapsed_time
println("\n=== 总体运行时间统计 ===")
println("模型1运行时间: $(round(elapsed_time, digits=2)) 秒")
println("模型2定义时间: $(round(model2_elapsed_time, digits=2)) 秒")
println("轨迹绘制时间: $(round(plot_elapsed_time, digits=2)) 秒")
println("KS检验分析时间: $(round(ks_elapsed_time, digits=2)) 秒")
println("总运行时间: $(round(total_elapsed_time, digits=2)) 秒")
println("总运行时间: $(round(total_elapsed_time/60, digits=2)) 分钟")



######
######
#########KS Test部分结束##########


#########