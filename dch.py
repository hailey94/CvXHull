import time
import torch
import numpy as np
import random
from tqdm import tqdm

class DCH():
    
    '''
    DCH Learns a dictionary for an input dataset using the distance from 
    convex-hull algorithm.

   INPUT:
   P             - the input dataset as a matrix with columns as 
                   datapoints and rows as dimensions
   epsilon       - error tolerance for each datapoint (optional)
   iterations    - no. of iterations to compute distance to convex hull
                   (optional)
   method_id     - numeric value specifying the method used to compute 
                   distance to convex hull
   stopping_func - either @max or @mean which is to be used as the
                   stopping criterion (default = @max)
   
   OUTPUT:
   U              - the dictionary learned by the algorithm as a matrix
   dist_array     - distance of farthest point from the convex-hull of the
                    dictionary at each iteration of the algorithm
   avg_dist_array - average distance of points from the convex-hull of the
                    dictionary at each iteration of the algorithm
   count_inactive - number of points within distance epsilon from the
                    convex-hull of the dictionary at each iteration
   
   TODO:
   Modify this code to return the sparse code in addition to the learned
   dictionary.

    '''
    def __init__(self, P, epsilon, iterations, stopping_func, cuda=1):
        self.P = P
        self.epsilon = epsilon
        self.iterations = iterations
        [d, n] = self.P.shape
        
        self.stopping_func = stopping_func
        
        if cuda:
            self.U = torch.zeros([d, 1]).cuda();
            self.dist_array = torch.zeros([1, n]).cuda();
            self.avg_dist_array = torch.zeros([1, n]).cuda();
            self.count_inactive = torch.zeros([1, n]).cuda();
        else:            
            self.U = torch.zeros([d, 1]);
            self.dist_array = torch.zeros([1, n]);
            self.avg_dist_array = torch.zeros([1, n]);
            self.count_inactive = torch.zeros([1, n]);

    
    def normc(self, m):
        
        ''' %NORMC Normalize columns of a matrix.'''
        
        [mr,mc] = m.shape
        if torch.cuda.is_available():
            if (mr == 1):
                m = torch.ones([1,mc]).cuda()
            else:
                m1 = torch.sqrt(1/torch.sum(m*m, dim=0))
                if len(m1.shape) == 1:
                    m1 = torch.unsqueeze(m1,dim=0)
                m = torch.matmul(torch.ones([mr,1]).cuda(), m1) * m
        else:
            if (mr == 1):
                m = torch.ones([1,mc])
            else:
                m = torch.matmul(torch.ones([mr,1]), torch.sqrt(1/torch.sum(m*m, dim=0))) * m
        del m1, mr, mc
        return m
    
    
    def compute_dist_point_to_line(self, q, a, b):

        '''
        COMPUTE_DIST_POINT_TO_LINE Calculates the distance from a query point (or
        a set of points) to a line segment (or a set of line segments). Also 
        returns the closest point to query on the line segment.

        There are two ways to use this function:
        1) q is a matrix and a and b are vectors. This function returns the
        set of distances for each point in q to the line segment ab.
        2) q is a matrix and a and b are matrices (q, a, and b have the same
        size). This function returns the distance of each point in q to the
        corresponding line segment from ab. For example, dist(i) = distance
        from q(:, i) to line segment a(:, i)b(:, i).

        INUPT:
        q - query point (or matrix of points) (dxn)
        a - end point of line segment (vector dx1)
        b - end point of line segment (vector dx1)

        OUTPUT:
        p    - point on ab which is closest to q (matrix dxn)
        dist - distance from q to to line segment ab (vector 1xn)
        '''
        if (a.shape[1]  > 1) & (b.shape[1]  > 1):
            x = q - a; # x is (dxn)
            y = b - a; # y is (dxn)
            
            t = torch.sum(x*y, dim=0); # t is (1xn)
            den = torch.sum(y*y, dim=0); # den is (1xn)
            
            nonzero_index = torch.nonzero(den);
            t[nonzero_index] = t[nonzero_index]/den[nonzero_index];
            p = a + y*t # p is (dxn)
            
            q = q.to(device='cuda:1')
            p = p.to(device='cuda:1')

            dist = torch.sqrt(torch.sum((q - p)*(q - p), dim=0)); # dist is (1xn)

        elif (a.shape[1] == 1) & (b.shape[1] == 1):

            if (a == b).all():
                p = a;
                dist = torch.cdist(a.T, q.T); # dist is (1xn)
            else:
                x = q - a; # x is (dxn)
                y = b - a; # y is (dx1)

                t = (x*y)/(y*y); # t is (nx1)
                
                p = y*t + a; # p is (dxn)
                
                q = q.to(device='cuda:1')
                p = p.to(device='cuda:1')

                dist = torch.sqrt(torch.sum((q - p)*(q - p), dim=0)); # dist is (1xn)
        return p, dist
    
    
    def compute_dist_to_chull(self,P, q):
        ''''
         COMPUTE_DISTANCE_CHULL Calculates the distance of a set of query points 
         to the convex hull of a set of points.

        INPUT: 
        P     - matrix of the set of points (as column vectors) which form the
               convex hull #self.U
        q     - matrix of query points represented as column vectors
        niter - no. of iterations to get approximate distance to convex hull
               N.B.: note that niter here reflects the sparsity of
               reconstruction #self.P

        OUTPUT:
        dist     - the vector of evaluated distances (1xn)
        t        - the matrix of reconstructed points (dxn)
        atom_idx - matrix of dictionary atoms (indicies) chosen during the
                  sparse c
        '''

        # Initial condition - find points t in P which are closest to query points 
        [_, n] = q.shape #  No. of query points
        atom_idx = torch.zeros([self.iterations, n])
        if len(P.shape) == 1:
            P = torch.unsqueeze(P, dim=1)

        [dist, min_index] = torch.min(torch.cdist(P.t().float(), q.t().float(), p=2), dim=0);
        t = P[:, min_index]
        
        atom_idx[1, :] = torch.zeros([n], dtype=torch.uint8)
        
        if len(t.shape) == 1:
            t = torch.unsqueeze(t, dim=1)
            
        # Compute the closest point from q to the convex hull of P
        # dist = zeros(1, n); % Vector of distances from q to conv(P)
        for i in range(0, self.iterations-1):
            v = q - t
            v = self.normc(v);
            vtx = torch.matmul(v.t(),P)
            [_, max_index] = torch.max(torch.matmul(v.t(),P), dim=1);
            
            p = P[:, max_index.t()]
            [t, dist] = self.compute_dist_point_to_line(q, t, p)
            t = t.to(device='cuda:0')
            atom_idx[i+1, :] = max_index.t()
            

        return dist, t, atom_idx
    
    def calculate_chull(self):
        start = time.time()
        [d, n] = self.P.shape

         # Learn the dictionary using the greedy distance to convex-hull algorithm
        r = random.randint(0, n-1)
        
        self.U[:, 0] = self.P[:, r]
        self.P = torch.cat([self.P[:,:r], self.P[:, r+1:]], dim=-1)
        
        if self.P.shape[1] == 0:
            return 0
        
        D, t, atom_idx = self.compute_dist_to_chull(self.U, self.P);
        [max_dist, max_index] = torch.max(D), torch.argmax(D);
        
        self.dist_array[:,0] = max_dist
        self.avg_dist_array[:,0] = torch.mean(D)
        self.count_inactive[:,0] = torch.sum(D <= self.epsilon) + 1

        flag = 0;
        for i in range(1, n):
            
            if i % 1000 == 0:
                print('max(D) on %d-th iteration: %.4f' %(i, torch.max(D).item()))
            
            #if max_dist <= epsilon
            if (self.stopping_func == 1):
                if torch.max(D) <= self.epsilon:
                    flag = 1;
                    # print('end of iteration on ', i+1)
                    break
            elif (self.stopping_func == 2):
                if torch.mean(D) <= self.epsilon:
                    # print('end of iteration on ', i+1)

                    flag = 1;
                    break
            else:
                print('should select max(1) or mean(2) as stopping function')
            
            if len(self.P[:, max_index].shape) == 1:
                append_p = torch.unsqueeze(self.P[:, max_index],dim=1)
            
            self.U = torch.cat((self.U, append_p), dim=1)
            self.P = torch.cat([self.P[:,:max_index], self.P[:, max_index+1:]], dim=-1)
            
            if i == n-1:
                flag = 1
                break
            
            D, t, atom_idx = self.compute_dist_to_chull(self.U, self.P);
            [max_dist, max_index] = torch.max(D), torch.argmax(D);
            self.dist_array[:,i] = max_dist;
            self.avg_dist_array[:,i] = torch.mean(D)
            self.count_inactive[:,i]= torch.sum(D <= self.epsilon) + i

        if flag == 1:
            self.U = self.U
            self.dist_array = self.dist_array
            self.avg_dist_array = self.avg_dist_array
            self.count_inactive = self.count_inactive

        time_taken = time.time() - start
        
        # print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time_taken)))
        return t

