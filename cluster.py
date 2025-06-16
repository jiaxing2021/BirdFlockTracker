

from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt


def cluster(points,p_flag=False):
    # Step 1: Cluster using DBSCAN
    clustering = DBSCAN(eps=300, min_samples=3).fit(points)
    labels = clustering.labels_
    clusters = [points[labels == label] for label in set(labels) if label != -1]

    # Step 2: Select the largest cluster
    main_cluster = max(clusters, key=lambda x: len(x)) if clusters else points

    # # Step 3: Remove outliers using Mahalanobis distance
    # if len(main_cluster) >= 2:
    #     mcd = MinCovDet().fit(main_cluster)
    #     distances = mcd.mahalanobis(main_cluster)
    #     threshold = np.percentile(distances, 90)
    #     cleaned_points = main_cluster[distances < threshold]
    # else:
    #     cleaned_points = main_cluster

    cleaned_points = main_cluster

    # Step 4: Plotting (All labels in English)

    if p_flag == True:
        plt.figure(figsize=(16, 5))

        # Plot 1: Original Points
        plt.subplot(1, 3, 1)
        plt.scatter(points[:, 0], points[:, 1], c='black', s=50)
        plt.title("Original Bird Positions")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)

        # Plot 2: Clustering Result
        plt.subplot(1, 3, 2)
        for label in set(labels):
            if label == -1:
                # Noise points in gray
                xy = points[labels == label]
                plt.scatter(xy[:, 0], xy[:, 1], c='gray', s=30, label="Noise")
            else:
                xy = points[labels == label]
                plt.scatter(xy[:, 0], xy[:, 1], s=50, label=f"Cluster {label}")
        plt.title("DBSCAN Clustering Result")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)

        # Plot 3: Cleaned Main Cluster
        plt.subplot(1, 3, 3)
        plt.scatter(cleaned_points[:, 0], cleaned_points[:, 1], c='green', s=60)
        plt.title("Cleaned Main Bird Group")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    return labels, cleaned_points