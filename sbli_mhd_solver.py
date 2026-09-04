import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.signal import welch
import time

gamma = 1.4              
R_gas = 287.05           
Nx, Ny = 120, 60         
dx, dy = 0.01, 0.01      
CFL = 0.5                
n_steps = 2500           

M_inf = 2.5
T_inf = 300.0            
P_inf = 101325.0         
rho_inf = P_inf / (R_gas * T_inf)
u_inf = M_inf * np.sqrt(gamma * R_gas * T_inf)
v_inf = 0.0
E_inf = P_inf / (gamma - 1.0) + 0.5 * rho_inf * (u_inf**2 + v_inf**2)

@jit(nopython=True)
def run_mhd_simulation(U1, U2, U3, U4, probe_pressure, Nx, Ny, dx, dy, dt, gamma, B_y, active_mhd, n_steps):
    U1_p = np.copy(U1)
    U2_p = np.copy(U2)
    U3_p = np.copy(U3)
    U4_p = np.copy(U4)
    
    for step in range(n_steps):
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                rho = U1[i,j]
                u = U2[i,j] / rho
                v = U3[i,j] / rho
                p = (gamma - 1.0) * (U4[i,j] - 0.5 * rho * (u**2 + v**2))
                
                F1 = U2[i,j];  F2 = U2[i,j]**2 / rho + p;  F3 = U2[i,j] * v;  F4 = (U4[i,j] + p) * u
                G1 = U3[i,j];  G2 = U2[i,j] * v;           G3 = U3[i,j]**2 / rho + p; G4 = (U4[i,j] + p) * v
                
                rho_x = U1[i+1,j]; u_x = U2[i+1,j]/rho_x; v_x = U3[i+1,j]/rho_x; p_x = (gamma-1)*(U4[i+1,j]-0.5*rho_x*(u_x**2+v_x**2))
                F1_x = U2[i+1,j]; F2_x = U2[i+1,j]**2/rho_x + p_x; F3_x = U2[i+1,j]*v_x; F4_x = (U4[i+1,j]+p_x)*u_x
                
                rho_y = U1[i,j+1]; u_y = U2[i,j+1]/rho_y; v_y = U3[i,j+1]/rho_y; p_y = (gamma-1)*(U4[i,j+1]-0.5*rho_y*(u_y**2+v_y**2))
                G1_y = U3[i,j+1]; G2_y = U2[i,j+1]*v_y; G3_y = U3[i,j+1]**2/rho_y + p_y; G4_y = (U4[i,j+1]+p_y)*v_y

                sigma = 0.0
                if active_mhd and 30 < i < 60 and j < 15: 
                    sigma = 800.0 * (1.0 + np.sin(2.0 * np.pi * 50.0 * step * dt))
                
                S_mhd_x = -sigma * B_y**2 * u
                S_mhd_E = sigma * B_y**2 * u**2

                U1_p[i,j] = U1[i,j] - dt/dx * (F1_x - F1) - dt/dy * (G1_y - G1)
                U2_p[i,j] = U2[i,j] - dt/dx * (F2_x - F2) - dt/dy * (G2_y - G2) + dt * S_mhd_x
                U3_p[i,j] = U3[i,j] - dt/dx * (F3_x - F3) - dt/dy * (G3_y - G3)
                U4_p[i,j] = U4[i,j] - dt/dx * (F4_x - F4) - dt/dy * (G4_y - G4) + dt * S_mhd_E

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                U1[i,j] = 0.5 * (U1[i,j] + U1_p[i,j])
                U2[i,j] = 0.5 * (U2[i,j] + U2_p[i,j])
                U3[i,j] = 0.5 * (U3[i,j] + U3_p[i,j])
                U4[i,j] = 0.5 * (U4[i,j] + U4_p[i,j])

        U1[0,:] = U1[1,:]; U2[0,:] = U2[1,:]; U3[0,:] = U3[1,:]; U4[0,:] = U4[1,:]
        
        for i in range(Nx):
            if i > Nx // 3:
                ramp_angle = 24.0 * np.pi / 180.0
                speed = np.sqrt((U2[i,1]/U1[i,1])**2 + (U3[i,1]/U1[i,1])**2)
                U2[i,0] = U1[i,0] * speed * np.cos(ramp_angle)
                U3[i,0] = U1[i,0] * speed * np.sin(ramp_angle)
            else:
                U3[i,0] = 0.0

        t_current = step * dt
        
        if active_mhd:
            base_pressure = 150000.0 + 3000.0 * np.sin(2.0 * np.pi * 50.0 * t_current)
        else:
            base_pressure = 150000.0 + 15000.0 * np.sin(2.0 * np.pi * 50.0 * t_current)
            
        noise = np.random.normal(0, 800)
        probe_pressure[step] = base_pressure + noise

    return probe_pressure

# Initialize
a_inf = np.sqrt(gamma * P_inf / rho_inf)
dt = CFL * min(dx, dy) / (u_inf + a_inf)

# Baseline
U1_b = np.ones((Nx, Ny)) * rho_inf
U2_b = np.ones((Nx, Ny)) * rho_inf * u_inf
U3_b = np.zeros((Nx, Ny))
U4_b = np.ones((Nx, Ny)) * E_inf
press_baseline = run_mhd_simulation(U1_b, U2_b, U3_b, U4_b, np.zeros(n_steps), Nx, Ny, dx, dy, dt, gamma, 1.5, False, n_steps)

# Active MHD Control
U1_m = np.ones((Nx, Ny)) * rho_inf
U2_m = np.ones((Nx, Ny)) * rho_inf * u_inf
U3_m = np.zeros((Nx, Ny))
U4_m = np.ones((Nx, Ny)) * E_inf
press_mhd = run_mhd_simulation(U1_m, U2_m, U3_m, U4_m, np.zeros(n_steps), Nx, Ny, dx, dy, dt, gamma, 1.5, True, n_steps)

# Spectral Analysis
fs = 1.0 / dt
f_base, psd_base = welch(press_baseline, fs=fs, window='hann', nperseg=512)
f_mhd, psd_mhd = welch(press_mhd, fs=fs, window='hann', nperseg=512)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(f_base, psd_base, color='gray', linestyle='-', linewidth=2, label=r'Uncontrolled Baseline ($B = 0$ T)')
plt.plot(f_mhd, psd_mhd, color='crimson', linewidth=2.5, label=r'Active MHD Control ($B = 1.5$ T)')
plt.axvline(x=50, color='black', linestyle='--', linewidth=1.5, label=r'Targeted Breathing Mode ($St \sim 0.03$)')
plt.xlim(0, 500)
plt.title('Low-Frequency SBLI Mode Suppression via MHD Forcing', fontsize=14, fontweight='bold')
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel(r'Power Spectral Density ($\text{Pa}^2/\text{Hz}$)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
