function dlqr(A,B,C,G,Q,R,N)
    mx = size(A)[2]
    mu = size(B)[2]
    mλ = size(C)[2]
    Ku = [[zeros(1,size(Q)[1]) for j=1:mu] for i=1:N-1]
    Kλ = [[zeros(1,size(Q)[1]) for j=1:mλ] for i=1:N-1]
    Pk = Q

    k = 0
    for outer k=N-1:-1:1
        D = B - C/(G*C)*G*B
        M11 = R + D'*Pk*B
        M12 = D'*Pk*C
        M21 = G*B
        M22 = G*C

        M = [M11 M12;M21 M22]
        b = [D'*Pk;G]*A

        Kk = M\b

        for i=1:mu
            Ku[k][i] = Kk[i:i,:]
        end

        Kuk = Kk[1:mu,:]
        Kλk = Kk[mu+1:mu+mλ,:]

        Abar = A-B*Kuk-C*Kλk
        Pkp1 = Q + Kuk'*R*Kuk + Abar'*Pk*Abar

        if norm(Pk-Pkp1) < 1e-5
            break
        end

        Pk = Pkp1
    end

    for k2=k-1:-1:1
        Ku[k2] = Ku[k2+1]
    end

    return Ku
end