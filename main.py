import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
from functools import partial
import sys
import time
import copy
import scipy.sparse as sp
from numba import njit, prange
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('TkAgg')


import Brownian_Dynamics_Functions as BDF



@njit(parallel=True, fastmath=True)
def calc_force_numba(r_vectors, L, a, repulsion_strength, debye_length, meg):
  '''
  This function computes the force between two blobs
  with vector between blob centers r, as well as the bouyant force -meg*z_hat, 
  and the force between blobs and the wall where h is the height of the blobs

  In this example the force is derived from the potential

  U(r) = U0 + U0 * (2*a-r)/b   if r<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  if r>=2*a,

  U(h) = U0 + U0 * (a-h)/b   if h<a
  U(h) = U0 * exp(-(h-a)/b)  if h>=a,

  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  force = np.zeros((N, 3))
  # loop over all particles
  for i in prange(N):
    force[i,2] -= meg
    h = r_vectors[i,2]
    if h > a:
        force[i,2] += (repulsion_strength / debye_length) * np.exp(-(h-a)/debye_length)
    else:
        force[i,2] += (repulsion_strength / debye_length)
    # loop over all other particles
    for j in range(N):
      if i == j:
        continue

      # caalculate periodic displacement vector between r_i and r_j
      dr = np.zeros(3)
      for k in range(3):
        dr[k] = r_vectors[j,k] - r_vectors[i,k]
        if L[k] > 0:
          dr[k] -= int(dr[k] / L[k] + 0.5 * (int(dr[k]>0) - int(dr[k]<0))) * L[k]

      # Compute force
      r_norm = np.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
      #####################
      for k in range(3):
        offset = 2*a
        if r_norm > (offset):
          force[i,k] += -((repulsion_strength / debye_length) * np.exp(-(r_norm-(offset)) / debye_length) / np.maximum(r_norm, 1.0e-12)) * dr[k]
        else:
          force[i,k] += -((repulsion_strength / debye_length) / np.maximum(r_norm, 1.0e-12)) * dr[k]

  return force.flatten()

def compute_deterministic_velocity_and_torque(r_vecs,Omega_vecs,M_rr,M_rt,M_tr, M_tt,force_calc):

        '''
        Compute the torque on bodies rotating with a prescribed
        angular velocity and subject to forces, i.e., solve the
        linear system
        
        M_rr * T = omega - M_rt * forces
        
        Then compute the translational velocity

        v = M_tr * T + M_tt * forces
        
        It returns the velocities and torques (v,T).
        '''
        # Compute forces
        force = force_calc(r_vecs)
        # Use constraint motion or free kinematics
        

        # Set RHS = omega - M_rt * force 
        RHS = Omega_vecs - M_rt(force, r_vecs)

        # Set linear operator 
        system_size = len(force)
        
        M_rr_mult = lambda f: M_rr(f,r_vecs)
        A = spla.LinearOperator((system_size, system_size), matvec = M_rr_mult, dtype='float64')

        # Scale RHS to norm 1
        RHS_norm = np.linalg.norm(RHS)
        if RHS_norm > 0:
            RHS = RHS / RHS_norm

        # Solve linear system 
        #counter = gmres_counter(print_residual = self.print_residual)
        (torque, info_precond) = spla.gmres(A,                        
                                                    RHS, 
                                                    x0=None, 
                                                    tol=1.0e-4, 
                                                    maxiter=1000)
                                                    

        # Scale solution with RHS norm
        if RHS_norm > 0:
            torque *= RHS_norm

        # Compute linear velocity
        velocity  = M_tt(force, r_vecs) + M_tr(torque, r_vecs) #multiply force/torque by Mobility to get linear velocities
        
        # Return linear velocity and torque
        return velocity, torque


if __name__ == "__main__":
    #physical units
    # unit system is : (Length scale- um, Force- pN, Energy scale- aJ, Time scale- s, Mass - mg)
    
    a = 0.66 # particle radius, um
    D = 2.0*a # particle diameter, um
    eta = 1.0e-3 #Pa.s
    Lx = 1.05*50.0*D # periodic length x direction (um) - NOTE: this implictly specifies the number of particles
    Ly = -1.0 #  periodic length y direction : negative numbers imply no periodicity
    Lz = -1.0 #  periodic length z direction : negative numbers imply no periodicity
    kbT = 0.004142 # (aJ ; atto-> e-18) where kb*T uses T = 25 C: this is our unit of energy
    L = np.array([Lx, Ly, Lz]) #L vector
    # make initial positions of particles 
    # format: [x_1, y_1, z_1, x_2, y_2, z_2, ...]

    # force parameters
    debye_length = 0.1*a # debeye length (um)
    rep_strength = 4.0*kbT # repulsion strength (aJ)
    g = 1*kbT # gravitational force (pN) = \Delta \rho * Volume * (9.81 m/s/s) 

    # create a forcing function from paramters
    force_calc  = partial(calc_force_numba
                               ,L=L
                               ,a=a
                               ,repulsion_strength=rep_strength
                               ,debye_length = debye_length
                               ,meg=g) 

    # create mobility functions
    Mtt_dot = partial(BDF.single_wall_mobility_trans_times_force_numba,eta=eta,a=a,L=L) # a function of positions: takes forces and gives linear velocities
    Mtr_dot = partial(BDF.single_wall_mobility_trans_times_torque_numba,eta=eta,a=a,L=L) # a function of positions: takes torques and gives linear velocities
    Mrt_dot = partial(BDF.single_wall_mobility_rot_times_force_numba,eta=eta,a=a,L=L) # a function of positions: takes forces and gives angular velocities
    Mrr_dot = partial(BDF.single_wall_mobility_rot_times_torque_numba,eta=eta,a=a,L=L) # a function of positions: takes torques and gives angular velocities

    # pick time step based on the fastest dimensionless timescale (or playing around)
    dt = 5e-3
    
    #initial conditions for the particles
    z_h = a + kbT/g
    x = np.arange(a,Lx,1.05*D) # 5 particles
    y1 = 4*a + 0*x # first wave of particles
    y2 = 4*a + 10*1.5*a + 0*x # second wave of particles
    z = a + 0*x

    r_vecs_part1 = np.column_stack((x,y1,z))
    r_vecs_part2 = np.column_stack((x,y2,z))
    r_vecs  = np.vstack((r_vecs_part1,r_vecs_part2)) #position of all the particles in the grid
    Np = len(r_vecs.ravel()) // 3 # the number of particles
    print(Np)
 
    # set the desired frequency for the rollers to spin at
    freq = 3 # Hz
    Omega = np.zeros(r_vecs.shape)
    Omega[:,0] = -freq*2.0*np.pi
    Omega = Omega.ravel()
    

    # a function to calculate the deterministic, linear velocities from the particle forces and the specified freq. (Hz)
    vel_torq_calc = partial(compute_deterministic_velocity_and_torque
                                          ,Omega_vecs = Omega
                                          ,M_rr = Mrr_dot
                                          ,M_rt = Mrt_dot
                                          ,M_tr = Mtr_dot
                                          ,M_tt = Mtt_dot
                                          ,force_calc=force_calc)
    #start time loop
    T_final = 100.0 # Final time (s)
    N_steps = int(np.round(T_final/dt)) # number of steps

    vel_p = 0*r_vecs
    plt.figure(figsize=(12, 8))
    max_z = 4.0*a
    norm = plt.Normalize(vmin=a, vmax=max_z)  # Normalize the z values for the color map within the range [a, 6*a]
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # This is needed for the color bar

    for i in range(N_steps):
      print(i)

      t0 = time.time()
      #run the deterministic velocity & torque calculation
      velocities, torq = vel_torq_calc(r_vecs)
      velocities = velocities.reshape((Np,3)) 
      #Integrate 
      for k in range(Np): #loop over particles
          for j in range(3): #loop over axis
              r_vecs[k,j] += dt * (1.5*velocities[k,j] - 0.5*vel_p[k,j]) #gives the updated position coordinates 
              vel_p = velocities
              if L[j]>0: #if periodic in that dimension
                  while r_vecs[k,j] > L[j]:
                    r_vecs[k,j] -=  L[j]
                  while r_vecs[k,j] < 0:
                    r_vecs[k,j] +=  L[j]

      dt1 = time.time() - t0
      #save data for this time step
      print(("Walltime for time step : %s" % dt1))

      # Generate scatter plot every n steps
      if i%300 == 0:

        plt.clf()  # Clear the figure
        x = r_vecs[:, 0]
        y = r_vecs[:, 1]
        z = r_vecs[:, 2]
        
        mean_y = np.mean(y)

        # Plot circles at each x, y position
        for xi, yi, zi in zip(x, y, z):
            color = plt.cm.viridis(norm(zi)) if a <= zi <= max_z else plt.cm.viridis(norm(a if zi < a else max_z))
            circle = plt.Circle((xi, yi), radius=a, color=color, ec=color)
            #plt.gca().add_patch(circle)
            plt.gca().add_patch(circle)
        
        # Add colorbar with fixed caxis
        plt.colorbar(sm, label='Height (um)')
        plt.xlim([0, Lx])
        plt.ylim([mean_y-40, mean_y+80])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title(f'Scatter plot of particle positions at time = {i*dt} s')
        plt.pause(0.01)
        plt.show(block=False)

        
        
    
    
        














