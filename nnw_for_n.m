function nnw_for_n( inp )

% Константы
cnt_of_hidden_layers = 5;       % Кол-во скрытых слоев
cnt_hid_n = 5;                  % кол-во нейронов в них
E_SUM_MAX = 5e+3;               % Максимальное допустимое значение функции ошибки
n = 10;                         % Кол-во цифр
m = 3;                          % Кол-во вариантов каждой цифры
eps = 1;
betta = 1;
alpha = 1;
max_epochs = 100;

X = imread([inp '.tif']);
im_sz = size(X);
S = im_sz(1) * im_sz(2);            % длина вектора и кол-во нейронов в одном слое
w = im_sz(2);
h = im_sz(1);

% Приводим X к вектору в биполярной форме
X = reshape(((X).* 2 - 1).', 1, S);

% Задаем случайную матрицу весов для скрытыв слоев
W_hidden = randi([1 99], cnt_hid_n, cnt_hid_n, cnt_of_hidden_layers - 1, n) * 0.01;
% И две матрицы для входа и выхода в скрытые слои
W_to_hidden = randi([1 99], S, cnt_hid_n, n) * 0.01;
W_from_hidden = randi([1 99], cnt_hid_n, S, n) * 0.01;

% матрица всех значений всех нейронов во всей сети (для ошибки)
Neuro = zeros(max(S, cnt_hid_n), 2 + cnt_of_hidden_layers, n);

% Создаем матрицу для хранения всех локальных градиентов
Local_Gradient = Neuro;

% Читаем изображения цифр
voc = cell(m,n);
for i = 1:m
    for j = '0':'9'
        voc{i,str2double(j)+1} = imread([j num2str(i) '.tif']);
        %     imfinfo([i '.tif'])
    end
end
% преобразуем их в биполярные вектора
Learning = cell(m,n);
for j = 1:m
    for i = 1:n
        Learning{j,i} = reshape((voc{j,i}.* 2 - 1).', 1, S);
    end
end
E = zeros(m,n);
E_W = zeros(1,10);
% Обучение методом обратного распространиения ошибки
tic
t = 0;

while t < max_epochs
    for L = 1:n
        for M = 1:m
            for Q = 1:4
                % Выбираем вектор из обучающего множества и подаем на вход
                O_k = Learning{M,L};
                X_out = zeros(1,cnt_hid_n);
                for p = 1:length(O_k)
                    Neuro(p,1, L) = O_k(p);
                end
                X_in = O_k;
                for i = 1:cnt_hid_n
                    summ = 0;
                    for j = 1:S
                        summ = summ + X_in(j) * W_to_hidden(j,i, L);
                    end
                    X_out(i) = alpha * tanh(-betta * summ);
                end
                for p = 1:length(X_out)
                    Neuro(p,2, L) = X_out(p);
                end
                for C = 1:cnt_of_hidden_layers - 1
                    X_in = X_out;
                    for i = 1:cnt_hid_n
                        summ = 0;
                        for j = 1:cnt_hid_n
                            summ = summ + X_in(j) * W_hidden(j,i,C, L);
                        end
                        X_out(i) = alpha * tanh(-betta * summ);
                    end
                    for p = 1:length(X_out)
                        Neuro(p,C + 2, L) = X_out(p);
                    end
                end
                X_out = zeros(1,S);
                for i = 1:S
                    summ = 0;
                    for j = 1:cnt_hid_n
                        summ = summ + X_in(j) * W_from_hidden(j,i, L);
                    end
                    X_out(i) = alpha * tanh(-betta * summ);
                end
                for p = 1:length(X_out)
                    Neuro(p,cnt_of_hidden_layers + 2, L) = X_out(p);
                end
                
                Y = X_out;
                Y = reshape(Y,w,h).';
                imshow(Y)
                Y = reshape(Y.', 1, S);
                
                E(M, L) = sum((Y - O_k).^2);
                
                
                % Корректировка весов методом обратного распространения
                % ошибки
                for i = 1:cnt_hid_n
                    y_i = Neuro(i, cnt_of_hidden_layers + 1, L);
                    for j = 1:S
                        Local_Gradient(j, cnt_of_hidden_layers + 2, L) = betta / alpha * (O_k(j) - Y(j)) * (alpha - Y(j)) * (alpha + Y(j));
                        delta_W_i_j = eps * Local_Gradient(j,cnt_of_hidden_layers + 2, L) * y_i;
                        W_from_hidden(i,j, L) = W_from_hidden(i,j, L) + delta_W_i_j;
                    end
                end
                for i = 1:cnt_hid_n
                    y_i = Neuro(i, cnt_of_hidden_layers, L);
                    for j = 1:cnt_hid_n
                        summ = 0;
                        for k = 1:S
                            summ = summ + W_from_hidden(j, k, L) * Local_Gradient(k, cnt_of_hidden_layers + 2, L);
                        end
                        Local_Gradient(j, cnt_of_hidden_layers + 1, L) = betta / alpha * (alpha - Neuro(j, cnt_of_hidden_layers, L)) * (alpha + Neuro(j, cnt_of_hidden_layers, L)) * summ;
                        delta_W_i_j = eps * Local_Gradient(j, cnt_of_hidden_layers + 1, L) * y_i;
                        W_hidden(i, j, cnt_of_hidden_layers - 1, L) = W_hidden(i, j, cnt_of_hidden_layers - 1, L) + delta_W_i_j;
                    end
                end
                if cnt_of_hidden_layers > 2
                    for p = cnt_of_hidden_layers - 1:-1:2
                        for i = 1:cnt_hid_n
                            y_i = Neuro(i, 1, L);
                            for j = 1:cnt_hid_n
                                summ = 0;
                                for k = 1:cnt_hid_n
                                    summ = summ + W_hidden(j, k, p, L) * Local_Gradient(k, p + 2);
                                end
                                Local_Gradient(j, p + 1, L) = betta / alpha * (alpha - Neuro(j, p, L)) * (alpha + Neuro(j, p, L)) * summ;
                                delta_W_i_j = eps * Local_Gradient(j, p + 1, L) * y_i;
                                W_hidden(i, j, p - 1, L) = W_hidden(i, j, p - 1, L) + delta_W_i_j;
                            end
                        end
                    end
                end
                for i = 1:S
                    y_i = Neuro(i, 1, L);
                    for j = 1:cnt_hid_n
                        summ = 0;
                        for k = 1:cnt_hid_n
                            summ = summ + W_hidden(j, k, 1, L) * Local_Gradient(k, 3);
                        end
                        Local_Gradient(j, 2, L) = betta / alpha * (alpha - Neuro(j, 1, L)) * (alpha + Neuro(j, 1, L)) * summ;
                        delta_W_i_j = eps * Local_Gradient(j, 2, L) * y_i;
                        W_to_hidden(i, j, L) = W_to_hidden(i, j, L) + delta_W_i_j;
                    end
                end
            end
        end
    end
    t = t + 1;
    disp(E)
    E_W(1, t) = sum(sum(E));
    disp(sum(sum(E)))
    % проверяем суммарную ошибку для окончания обучения
    if sum(sum(E)) < E_SUM_MAX
        disp('END: 5e+4')
        break;
    end
end
toc
disp(t)
close all
% Рисуем графики ошибки
pause(0.5)
subplot(1,3,1)
plot(E)
subplot(1,3,2)
bar(E_W)


% Проверка картинци, которую подал пользователь уже обученной сети
X_out = zeros(1,cnt_hid_n);
X_s = cell(1,n);
Miss = zeros(1,n);
    for N = 1:n
        X_in = X;
        for i = 1:cnt_hid_n
            summ = 0;
            for j = 1:S
                summ = summ + X_in(j) * W_to_hidden(j,i, N);
            end
            X_out(i) = alpha * tanh(-betta * summ);
        end
        
        for C = 1:cnt_of_hidden_layers - 1
            X_in = X_out;
            for i = 1:cnt_hid_n
                summ = 0;
                for j = 1:cnt_hid_n
                    summ = summ + X_in(j) * W_hidden(j,i,C, N);
                end
                X_out(i) = alpha * tanh(-betta * summ);
            end
        end
        X_out = zeros(1,S);
        for i = 1:S
            summ = 0;
            for j = 1:cnt_hid_n
                summ = summ + X_in(j) * W_from_hidden(j,i, N);
            end
            X_out(i) = alpha * tanh(-betta * summ);
        end
        X_s{1, N} = X_out;
        Miss(1, N) = sumabs(X_out - X);
    end
    [x, k] = min(Miss);
    Y = X_s{1, k};
    Y = reshape(Y,w,h).';
    subplot(1,3,3);
    subimage(Y);
end