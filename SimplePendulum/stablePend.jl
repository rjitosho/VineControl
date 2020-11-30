function stable_rollout(x0,f,dt,tf)
    Ku = [0.0 0.0 9.603461750376233 -1.9275002294663957 1.1802534931859052e-16 3.8550004589327913]
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    U = zeros(1,N-1)
    Lam = zeros(2,N-1)
    X[:,1] = x0
    for k = 1:N-1
        U[k] = (-Ku*(X[:,k]-xf))[1]
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam, U
end

xf = [.25*sqrt(2);.25*sqrt(2);3*pi/4;zeros(3)]
x1, _ = f(xf,[1.],dt)
X, Lam, U=stable_rollout(x1,f,dt,tf)
plot(X[3,:])