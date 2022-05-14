
max2 = max(n)
normalised_n=[]
for i in range(len(n)):
    value2 = n[i]
    normalised_n = np.append(normalised_n, (value2/max2))  # Normalisation
fit = p[0]*(bins_mean2**4) + p[1]*(bins_mean2**3) + p[2]*(bins_mean2**2) + p[3]*bins_mean2 + p[4]
    
plt.scatter(bins_mean2, normalised_n, s =5, color = 'r',label='Simulated Data')
plt.plot(bins_mean2,fit, color = 'lime',linewidth = 3,label='Real Fit')
plt.axvline(x=0, color = 'gray', ls = '--')
plt.title('Plot of Real Illumination Data')
plt.xlabel('Position from Centre [mm]')
plt.ylabel('Relative Intensity')
plt.legend()
plt.savefig('realdataplot.png',dpi = 1000)
plt.show()
np.savetxt("relative_intensity.csv", bins_mean2, delimiter=",")
np.savetxt("mm.csv", normalised_n, delimiter=",")
print('Real data fit: y =','{0:.3g}'.format(p[0]),'x^4  +','{0:.3g}'.format(p[1]),'x^3  +','{0:.3g}'.format(p[2]),'x^2 +','{0:.3g}'.format(p[3]),'x +','{0:.3g}'.format(p[4]))

# Values of merit
chi_list = ((fit-normalised_n)**2)/fit
chi_squared = sum(chi_list)
r_cs = chi_squared/99
print('Reduced Chi-Squared =','{0:.4g}'.format(r_cs))
