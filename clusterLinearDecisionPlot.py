import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

from CBP import GPSVM, PSVM
from PSVM import MagKmeans
from dataGeneration import *
from pathlib import Path
from sklearn import svm

tol = 0.03
LENGTH = 2

def plot_GPSVM(df,args,outfile):
	'''
	add a linear decision line for each cluster
	'''
	global tol, LENGTH
	GPSVM_model = GPSVM(clusterNum = args[0],ensembleNum=args[1],C = args[2])
	GPSVM_model.fit(df[['X','Y']].values, df['Label'].values)
	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(10, 10)
	fig.set_dpi(500)
	colors = cm.get_cmap('viridis', GPSVM_model.kmeans.n_clusters)
	ax.scatter(df[df['Label'] ==1]['X'], df[df['Label'] ==1]['Y'], c = 'grey', marker = '*' )
	ax.scatter(df[df['Label'] ==-1]['X'], df[df['Label'] ==-1]['Y'], c = 'black', s =  10)
	for c, l, svm  in zip(colors(np.unique(GPSVM_model.kmeans.labels_)),  
						 range(GPSVM_model.kmeans.n_clusters),
						 GPSVM_model.SVM):
		# pos_points = df.iloc[np.array(GPSVM_model.cbp.points)[GPSVM_model.kmeans.labels_ == l][:,0]]
		# neg_points = df.iloc[np.array(GPSVM_model.cbp.points)[GPSVM_model.kmeans.labels_ == l][:,1]]
		# print(500)
		# ax.scatter(pos_points['X'], pos_points['Y'], c = 'none', edgecolors=c,
		# 			marker = 'H', s = 500, label='Gabriel edited set' )
		# ax.scatter(neg_points['X'], neg_points['Y'], c = 'none', edgecolors=c,
		# 			marker = 'H', s = 100, label='Gabriel edited set' )

		points = df.iloc[  np.unique(np.array(GPSVM_model.cbp.points)[GPSVM_model.kmeans.labels_ == l].reshape(-1)) ]
		ax.scatter(points['X'], points['Y'], c = 'none', edgecolors=c,
					marker = 'H', s = 50 + l*10, label='Gabriel edited set' )
		# # straight lines
		center = GPSVM_model.clusterCentroids[l]

		w = svm.coef_[0]           # w consists of 2 elements
		b = svm.intercept_[0]      # b consists of 1 element  \
		# 0 = wX + b
		#w[0] and w[1] are close to 0, no decision boundary
		if abs(w[1]) < tol and abs(w[0]) < tol:
			ax.text(center[0],center[1],'No apparent\n linear d.b.', c = 'red')
		else:
			if abs(w[1]) < tol:
				#horizontal
				yy = np.linspace(center[1] - 5, center[1] + 5, 200)
				xx = np.full_like(yy, -b/w[0])
				ax.text(center[0],center[1],str(l), c = 'red')

			elif abs(w[0]) < tol:
				#vertical
				xx = np.linspace(center[0] - 5, center[0] + 5, 200)
				yy = np.full_like(xx, -b/w[1])
				ax.text(center[0],center[1],str(l),c = 'red')

			else:
				xx = np.linspace(center[0] - 5, center[0] + 5, 200)
				yy = -(w[0] / w[1]) * xx - b / w[1]
				ax.text(center[0],center[1],str(l),c = 'red')
			dToLine = abs(w[0]*center[0] + w[1]*center[1] + b)/np.sqrt(w[0]**2 + w[1]**2)
			dToCenter= np.sqrt((yy -  center[1] )**2 + (xx - center[0])**2)
			index = (dToCenter**2 - dToLine**2)  < LENGTH
			ax.plot(xx[index], yy[index], c = c, linewidth = 3)
			#ax.plot(xx, yy, c = c, linewidth = 3)
		# ax.set_ylim(0,10)
		# ax.set_xlim(0,10)
		plt.savefig(Path(outfile), bbox_inches='tight', format = 'pdf')

	

def plot_PSVM(df,args,outfile):
	'''
	'''
	global tol, LENGTH
	PSVM_model = PSVM(clusterNum = args[0],ensembleNum=args[1],C = args[2], R = args[3], max_iterations = 500)
	PSVM_model.fit(df[['X','Y']].values, df['Label'].values)

	# Get cluster centroids and labels
	centroids = PSVM_model.MagKmeans.cluster_centers_
	predicted_labels = PSVM_model.MagKmeans.labels_
	# Create a scatter plot for each group
	colors = cm.get_cmap('viridis', PSVM_model.MagKmeans.K)
	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(10, 10)
	fig.set_dpi(500)
	# Define the significance level (alpha) and degrees of freedom (df)
	alpha = 0.2  # For a 95% confidence level
	degofFreedom = 2       # Degrees of freedom (can be adjusted)
	# Calculate the chi-squared critical value
	chi2_critical_value = stats.chi2.ppf(1 - alpha, degofFreedom)
	for c, l, svm  in zip(colors(np.unique(PSVM_model.MagKmeans.labels_)),  
						 range(PSVM_model.MagKmeans.K),
						 PSVM_model.SVM):
		condition_plus = (df['Label'] == 1) & (predicted_labels == l)
		condition_minus = (df['Label'] == -1) & (predicted_labels == l)
		ax.scatter(df[ condition_plus]['X'], df[condition_plus]['Y'], marker='*', facecolors='none', label="point with label 1 in cluster" + str(l),  edgecolor = c, alpha=0.5) #
		ax.scatter(df[condition_minus]['X'], df[condition_minus]['Y'], marker='o', label="point with label -1 in cluster" + str(l), color = c,alpha = 0.7) 

		cluster_points = df[['X','Y']].values[predicted_labels == l]
		

		# # straight lines
		center = PSVM_model.clusterCentroids[l]
		w = svm.coef_[0]           # w consists of 2 elements
		b = svm.intercept_[0]      # b consists of 1 element  \
		
		# 0 = wX + b
		#w[0] and w[1] are close to 0, no decision boundary
		if abs(w[1]) < tol and abs(w[0]) < tol:
			ax.text(center[0],center[1],'No apparent\n linear d.b.', c = 'red')
		else:
			if abs(w[1]) < tol:
				#horizontal
				yy = np.linspace(center[1] - 5, center[1] + 5, 200)
				xx = np.full_like(yy, -b/w[0])
				ax.text(center[0],center[1],str(l), c = 'red')

			elif abs(w[0]) < tol:
				#vertical
				xx = np.linspace(center[0] - 5, center[0] + 5, 200)
				yy = np.full_like(xx, -b/w[1])
				ax.text(center[0],center[1],str(l),c = 'red')

			else:
				xx = np.linspace(center[0] - 5, center[0] + 5, 200)
				yy = -(w[0] / w[1]) * xx - b / w[1]
				ax.text(center[0],center[1],str(l),c = 'red')
			dToLine = abs(w[0]*center[0] + w[1]*center[1] + b)/np.sqrt(w[0]**2 + w[1]**2)
			dToCenter= np.sqrt((yy -  center[1] )**2 + (xx - center[0])**2)
			index = (dToCenter**2 - dToLine**2)  < LENGTH
			ax.plot(xx[index], yy[index], c = c, linewidth = 3)
			#ax.plot(xx, yy, c = c, linewidth = 3)
	# ax.set_ylim(0,10)
	# ax.set_xlim(0,10)
	# ax.set_xlabel('Feature 1')
	# ax.set_ylabel('Feature 2')
	#ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.savefig(Path(outfile), bbox_inches='tight', format = 'pdf')












def main():

	num_points = 500
	# df = create_binary_class_boundary_concave(num_points)
	# plot_GPSVM(df,[20, 1, 0.5],'Plots/GPSVM_concave.pdf')
	# plot_PSVM(df,[20, 1, 0.5, 5],'Plots/PSVM_concave.pdf')


	
	# df = create_binary_class_boundary_twoSquares(num_points)
	# plot_GPSVM(df,[5, 1, 0.5],'Plots/GPSVM_twoSquares.pdf')
	# plot_PSVM(df,[5, 1, 0.5, 5],'Plots/PSVM_twoSquares.pdf')

	

	df = create_binary_class_boundary_spiral(num_points)
	#plot_GPSVM(df,[20, 1, 0.5],'Plots/GPSVM_spiral.pdf')
	plot_PSVM(df,[20, 1, 0.5, 5],'Plots/PSVM_spiral.pdf')

if __name__ == '__main__':
	main()