import numpy as np
from numpy.linalg import norm  as F

class ACA():
    def __init__(self,Z, R, f, u, v, i, j,e):
        self.Z=Z
        self.R=R
        self.f=f
        self.u=u
        self.v=v
        self.i=i
        self.j=j
        self.e=e

    def solve(self):
        self.aca_init(self.Z, self.R, self.f, self.u, self.v, self.i, self.j)
        self.iter_k(self.e,self.Z, self.R, self.f, self.u, self.v, self.i, self.j)
        #print("solve_done")

    def F1(self,a):
        return F(a) ** 2

    def F2(self,u, v, k):
        return 2 * sum([np.dot(u[J], u[k]) * np.dot(v[J], v[k]) for J in range(k)])

    def order(self,a,o):
        for num in np.argsort(abs(a))[:: -1]:
            if num not in o:
                o.append(num)
                break

    def aca_init(self,Z, R, f, u, v, i, j):
        R[i[0], :] = Z[i[0], :]  # I1
        j.append(np.argmax(abs(R[i[0], :])))  # J1
        v_ = R[i[0], :] / R[i[0], j[0]].copy()
        v.append(v_)  # v1
        R[:, j[0]] = Z[:, j[0]]
        u_ = R[:, j[0]].copy()
        u.append(u_)  # u1
        f.append((0 + self.F1(u[0]) * self.F1(v[0])))


        self.order(R[:, j[0]], i)

    def iter_k(self,e, Z, R, f, u, v, i, j):
        k = 1
        while k < min(len(Z), len(Z[0])) - 1:
            R[i[k], :] = Z[i[k], :] - np.sum([np.dot(u[I][i[k]], v[I]) for I in range(k)], axis=0)

            self.order(R[i[k], :], j)
            v_ = R[i[k], :] / R[i[k], j[k]].copy()
            v.append(v_)

            R[:, j[k]] = Z[:, j[k]] - np.sum([np.dot(v[I][j[k]], u[I]) for I in range(k)], axis=0)
            u_ = R[:, j[k]].copy()
            u.append(u_)

            temp = self.F1(u[k]) * self.F1(v[k])

            f.append((f[k - 1] + temp + self.F2(u, v, k)))

            if temp <= (e ** 2) * f[k]:
                break

            self.order(R[:, j[k]], i)
            k += 1

    def __del__(self):
        pass
        #print("over")

