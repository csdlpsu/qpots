function [Xout, Yout, time_out] = TSEMO_run(func, X, Y, lb, ub, iters, batch_number)

    py.sys.path().append('/home/kade/work_soft/mobo/TS-EMO');
    f = str2func(func);
    opt = TSEMO_options;
    opt.maxeval = batch_number*iters;
    opt.NoOfBachSequential = batch_number;
    X = X;
    Y = zeros(size(X, 1), size(X, 2));     % corresponding matrix of response data
    for k = 1:size(X,1)
        X(k,:) = X(k,:).*(ub-lb)+lb;        % adjustment of bounds
        Y(k,:) = f(X(k,:));                 % calculation of response data
    end
    [Xpareto,Ypareto,X,Y,XparetoGP,YparetoGP,YparetoGPstd,hypf,times] = TSEMO_V4(f,X,Y,lb,ub,opt);
    Xout = X;
    Yout = Y;
    time_out = times;

end
