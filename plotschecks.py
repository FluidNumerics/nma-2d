# import the pandas library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('dark_background')
plt.style.use('seaborn-v0_8-whitegrid')
plt.switch_backend('agg')

# specify the file path or URL
file_path = '/home/joe/xnma/checks_2023-10-18-17-18/stommelGyre.py/stommelgyre-energy.csv'  # replace with your csv file path


# read the csv file using pandas
df = pd.read_csv(file_path)

df['Interior Energy (numerical)'] = df['interior divergent energy (numerical)'] + df['interior rotational energy (numerical)'] 
df['Boundary Energy (numerical)'] = df['boundary divergent energy (numerical)'] + df['boundary rotational energy (numerical)']

df['Interior Energy (exact)'] = df['interior divergent energy (exact)'] + df['interior rotational energy (exact)'] 
df['Boundary Energy (exact)'] = df['boundary divergent energy (exact)'] + df['boundary rotational energy (exact)']

df_40 = df[df['nx'] == 40]
plt.figure(figsize=(6.4, 4.8))
plt.plot( df_40['nmodes'], df_40['Interior Energy (numerical)'], '--o', label='Interior (numerical)', markersize=3, linewidth=1)
plt.plot( df_40['nmodes'], df_40['Boundary Energy (numerical)'], '--o', label='Boundary (numerical)', markersize=3, linewidth=1)
plt.plot( df_40['nmodes'], df_40['Interior Energy (numerical)']+df_40['Boundary Energy (numerical)'], '-o', label='Total (numerical)', markersize=3, linewidth=1.5)
plt.plot( df_40['nmodes'], df_40['Interior Energy (exact)'], '--x', label='Interior (exact)', markersize=4, linewidth=1)
plt.plot( df_40['nmodes'], df_40['Boundary Energy (exact)'], '--x', label='Boundary (exact)', markersize=4, linewidth=1)
plt.plot( df_40['nmodes'], df_40['Interior Energy (exact)']+df_40['Boundary Energy (exact)'], '-x', label='Total (exact)', markersize=4, linewidth=1.5)
plt.title( 'Mesh size 40x40 ')
plt.xlabel('N modes')
plt.ylabel('Energy')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#plt.grid(color='gray', linestyle='-', linewidth=0.5)
# display the plot
plt.savefig(f"{file_path}_nx40.png", bbox_inches='tight')


plt.figure(figsize=(6.4, 4.8))
plt.plot( df['nx'][0:3], df['Interior Energy (numerical)'][0:3], '--o', label='Interior (numerical)', markersize=3, linewidth=1)
plt.plot( df['nx'][0:3], df['Boundary Energy (numerical)'][0:3],'--o',label='Boundary (numerical)', markersize=3, linewidth=1)
plt.plot( df['nx'][0:3], df['Interior Energy (numerical)'][0:3]+df['Boundary Energy (numerical)'][0:3],'-o',label='Total (numerical)', markersize=3, linewidth=1.5)
plt.plot( df['nx'][0:3], df['Interior Energy (exact)'][0:3], '--x', label='Interior (exact)', markersize=4, linewidth=1)
plt.plot( df['nx'][0:3], df['Boundary Energy (exact)'][0:3], '--x', label='Boundary (exact)', markersize=4, linewidth=1)
plt.plot( df['nx'][0:3], df['Interior Energy (exact)'][0:3]+df['Boundary Energy (exact)'][0:3], '-x', label='Total (exact)', markersize=4, linewidth=1.5)
plt.title( '1/4 modes ')
plt.xlabel('nx')
plt.ylabel('Energy')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#plt.grid(color='gray', linestyle='-', linewidth=0.5)
# display the plot
plt.savefig(f"{file_path}_mp25.png", bbox_inches='tight')


plt.figure(figsize=(6.4, 4.8))
plt.plot( df['nx'][4:7], df['Interior Energy (numerical)'][4:7], '--o', label='Interior (numerical)', markersize=3, linewidth=1)
plt.plot( df['nx'][4:7], df['Boundary Energy (numerical)'][4:7],'--o',label='Boundary (numerical)', markersize=3, linewidth=1)
plt.plot( df['nx'][4:7], df['Interior Energy (numerical)'][4:7]+df['Boundary Energy (numerical)'][4:7],'-o',label='Total (numerical)', markersize=3, linewidth=1.5)
plt.plot( df['nx'][4:7], df['Interior Energy (exact)'][4:7], '--x', label='Interior (exact)', markersize=4, linewidth=1)
plt.plot( df['nx'][4:7], df['Boundary Energy (exact)'][4:7], '--x', label='Boundary (exact)', markersize=4, linewidth=1)
plt.plot( df['nx'][4:7], df['Interior Energy (exact)'][4:7]+df['Boundary Energy (exact)'][4:7], '-x', label='Total (exact)', markersize=4, linewidth=1.5)
plt.title( '1/2 modes ')
plt.xlabel('nx')
plt.ylabel('Energy')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#plt.grid(color='gray', linestyle='-', linewidth=0.5)
# display the plot
plt.savefig(f"{file_path}_mp50.png", bbox_inches='tight')


plt.figure(figsize=(6.4, 4.8))
plt.plot( df['nx'][8:11], df['Interior Energy (numerical)'][8:11], '--o', label='Interior (numerical)', markersize=3, linewidth=1)
plt.plot( df['nx'][8:11], df['Boundary Energy (numerical)'][8:11],'--o',label='Boundary (numerical)', markersize=3, linewidth=1)
plt.plot( df['nx'][8:11], df['Interior Energy (numerical)'][8:11]+df['Boundary Energy (numerical)'][8:11],'-o',label='Boundary (numerical)', markersize=3, linewidth=1.5)
plt.plot( df['nx'][8:11], df['Interior Energy (exact)'][8:11], '--x', label='Interior (exact)', markersize=4, linewidth=1)
plt.plot( df['nx'][8:11], df['Boundary Energy (exact)'][8:11], '--x', label='Boundary (exact)', markersize=4, linewidth=1)
plt.plot( df['nx'][8:11], df['Interior Energy (exact)'][8:11]+df['Boundary Energy (exact)'][8:11], '-x', label='Total (exact)', markersize=4, linewidth=1.5)
plt.title( 'All modes ')
plt.xlabel('nx')
plt.ylabel('Energy')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#plt.grid(color='gray', linestyle='-', linewidth=0.5)
# display the plot
plt.savefig(f"{file_path}_mp100.png", bbox_inches='tight')

#print(df.keys())
plt.figure(figsize=(6.4, 4.8))
plt.plot( df['nx'][0:3], df['eigenmode search runtime (s)'][0:3], '--o', label='1/4 modes', markersize=3, linewidth=1 )
plt.plot( df['nx'][4:7], df['eigenmode search runtime (s)'][4:7], '--o', label='1/2 modes', markersize=3, linewidth=1 )
plt.plot( df['nx'][8:11], df['eigenmode search runtime (s)'][8:11], '--o', label='All modes', markersize=3, linewidth=1 )
plt.title( 'Eigenpair search runtime')
plt.xlabel('nx')
plt.ylabel('Runtime (s)')
#plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# display the plot
plt.savefig(f"{file_path}_runtime.png", bbox_inches='tight')
