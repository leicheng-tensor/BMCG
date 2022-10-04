% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

function [G_U,G_V,m,n] = createGraphsFromMovieLens1m(users,movies)
% Import MovieLens 1M user and movie features
% import *.dat as string table "users" and "movies"


% User features
User = struct('user_id',double(users(:,1)),'gender',char(users(:,2)),'age',double(users(:,3)),'occupation',double(users(:,4)));
m = max(User.user_id);
% num_user = max(User.user_id);
User_feature_vec = [User.user_id,zeros(m,7),User.gender == 'F',zeros(m,20)];
UserAgeList = [1,18,25,35,45,50,56];
for i=1:7
    User_feature_vec(:,i+1) = (UserAgeList(i) == User.age);
end
for i=1:20
    User_feature_vec(:,i+9) = (i == User.occupation); % "other" as '0' is removed
end
% Movie features
Num_rating = size(movies,1);
for i = 1:Num_rating
    if ~ismissing(movies(i,4))
        movies(i,3) = movies(i,4);
    end
end
Movie = struct('movie_id',double(movies(:,1)),'genre',movies(:,3));
n = max(Movie.movie_id);
% num_movie = max(Movie.movie_id);
MovieGenreList = {'Action','Adventure','Animation',"Children's",'Comedy','Crime',...
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',...
    'Mystery','Romance','Sci-Fi','Thriller','War',',Western'};
a = randperm(n);
b = sort(a)';
Movie_feature_vec = [b,zeros(n,length(MovieGenreList))];
% MovieGenre = cellstr(Movie.genre);
for j = 1:Num_rating
    for i=1:length(MovieGenreList)
        id = Movie.movie_id(j);
        Movie_feature_vec(id,i+1) = sum(strcmp(MovieGenreList{i},strsplit(Movie.genre(j),'|')));
    end
end

    % kNN (k=10), Euclidean distance metric
    [knn_idx, knn_D] = knnsearch(User_feature_vec(:,2:end),User_feature_vec(:,2:end),'K',11,'IncludeTies',true);
    final_knn = zeros(m,10);
    % Solve situation of more than ten equally distant by randomly sampling
    for i =1:m
        self_loop_idx = find(knn_idx{i} == i);
        knn_idx{i}(self_loop_idx)=[];
        knn_D{i}(self_loop_idx)=[];
        cut_off_dist = knn_D{i}(10);
        below_cut_off_idx = find(knn_D{i} < cut_off_dist);
        on_cut_off_idx = find(knn_D{i}==cut_off_dist);
        on_cut_off_rand_selected = randperm(length(on_cut_off_idx),10-length(below_cut_off_idx));
        final_knn(i,:) = knn_idx{i}([below_cut_off_idx,on_cut_off_rand_selected]);
    end
    user_graph_tuple=[repelem(1:size(final_knn,1),10)',reshape(final_knn',size(final_knn,1)*size(final_knn,2),1),ones(size(final_knn,1)*10,1)];
    G_U_directed = sparse(user_graph_tuple(:,1),user_graph_tuple(:,2),user_graph_tuple(:,3),m,m);
    G_U = G_U_directed + G_U_directed';

    % kNN, k=10, Euclidean distance
    [knn_idx, knn_D] = knnsearch(Movie_feature_vec(:,2:end),Movie_feature_vec(:,2:end),'K',11,'IncludeTies',true);
    final_knn = zeros(n,10);
    for i=1:n
        self_loop_idx = find(knn_idx{i} == i);
        knn_idx{i}(self_loop_idx)=[];
        knn_D{i}(self_loop_idx)=[];
        cut_off_dist = knn_D{i}(10);
        below_cut_off_idx = find(knn_D{i} < cut_off_dist);
        on_cut_off_idx = find(knn_D{i}==cut_off_dist);
        on_cut_off_rand_selected = randperm(length(on_cut_off_idx),10-length(below_cut_off_idx));
        final_knn(i,:) = knn_idx{i}([below_cut_off_idx,on_cut_off_rand_selected]);
    end
    movie_graph_tuple=[repelem(1:size(final_knn,1),10)',reshape(final_knn',size(final_knn,1)*size(final_knn,2),1),ones(size(final_knn,1)*10,1)];
    G_V_directed = sparse(movie_graph_tuple(:,1),movie_graph_tuple(:,2),movie_graph_tuple(:,3),n,n);
    G_V = G_V_directed + G_V_directed';
end

