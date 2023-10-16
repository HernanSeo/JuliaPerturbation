function loglh = kf(y, TT, RQR, DD, ZZ, HH, t0)
    [nobs, ny] = size(y);
    [ns, ~] = size(TT);
    At = zeros(ns, 1);
    TT_old = TT;
    RQR_old = RQR;
    P_10_old = eye(size(TT));
    loglhvec = zeros(nobs, 1);
    P_10 = P_10_old;
    diferenz = 0.1;
    
    while diferenz > 1e-25
        P_10 = TT_old * P_10_old * TT_old' + RQR_old;
        diferenz = max(max(abs(P_10 - P_10_old)));
        RQR_old = TT_old * RQR_old * TT_old' + RQR_old;
        TT_old = TT_old * TT_old;
        P_10_old = P_10;
    end
    
    Pt = P_10;
    loglh = 0.0;
    yaux = zeros(size(DD));
    ZPZ = zeros(size(HH));
    TTPt = zeros(size(Pt));
    TAt = zeros(size(At));
    KiF = zeros(size(At));
    PtZZ = Pt * ZZ';
    Kt = PtZZ;
    TTPtTT = Pt;
    KFK = Pt;
    iFtnut = zeros(size(DD));
    
    for i = 1:nobs
        yaux = ZZ * At + DD;
        yhat = yaux + DD;
        nut = (y(i, :)' - yhat);
        PtZZ = Pt * ZZ';
        ZPZ = ZZ * PtZZ;
        Ft = ZPZ + HH;
        Ft = 0.5 * (Ft + Ft');
        dFt = det(Ft);
        iFt = inv(Ft);
        iFtnut = iFt * nut;
        loglhvec(i) = -0.5 * log(dFt) - 0.5 * iFtnut' * nut;
        TTPt = TT * Pt;
        Kt = TTPt * ZZ';
        TAt = TT * At;
        KiF = Kt * iFtnut;
        At = TAt + KiF;
        TTPtTT = TTPt * TT';
        KFK = Kt * (Ft \ Kt');
        Pt = TTPtTT - KFK + RQR;
    end
    loglh = sum(loglhvec) - nobs * 0.5 * ny * log(2 * pi);
end
