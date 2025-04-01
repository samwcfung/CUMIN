#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Fluorescence Analysis Module
-----------------------------------
Implements advanced analysis techniques:
1. Machine learning classification of ROI activity patterns
2. Correlation analysis between nearby ROIs
3. Earth Mover's Distance (EMD) calculation for pain models
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import ot  # For EMD (optimal transport)

def run_advanced_analysis(
    fluorescence_data, 
    roi_masks, 
    metrics_df, 
    output_dir, 
    config, 
    logger
):
    """
    Run advanced fluorescence analysis if enabled in configuration.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    output_dir : str
        Directory to save the analysis results
    config : dict
        Configuration parameters
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    dict or None
        Advanced analysis results if advanced analysis is enabled, None otherwise
    """
    # Check if advanced analysis is enabled
    if not config.get("advanced_analysis", {}).get("enabled", False):
        logger.info("Advanced analysis is disabled in configuration, skipping")
        return None
    
    # Create analyzer and run analysis
    analyzer = AdvancedFluoroAnalysis(config.get("advanced_analysis", {}), logger)
    results = analyzer.run_analysis(fluorescence_data, roi_masks, metrics_df, output_dir)
    
    return results

class AdvancedFluoroAnalysis:
    """Advanced fluorescence analysis methods."""
    
    def __init__(self, config, logger):
        """
        Initialize the advanced analysis module.
        
        Parameters
        ----------
        config : dict
            Configuration parameters for advanced analysis
        logger : logging.Logger
            Logger object
        """
        self.config = config
        self.logger = logger
        self.logger.info("Initializing advanced fluorescence analysis module")
        
        # Extract configuration parameters with defaults
        self.ml_enabled = config.get("ml_enabled", True)
        self.correlation_enabled = config.get("correlation_enabled", True)
        self.emd_enabled = config.get("emd_enabled", True)
        
        # ML parameters
        self.ml_config = config.get("ml", {})
        self.n_clusters = self.ml_config.get("n_clusters", 3)
        self.feature_selection = self.ml_config.get("feature_selection", ["peak_amplitude", "peak_rise_time", "std_df_f"]) #changed from "rise_time"
        
        # Correlation parameters
        self.corr_config = config.get("correlation", {})
        self.distance_threshold = self.corr_config.get("distance_threshold", 50)
        self.corr_method = self.corr_config.get("method", "pearson")
        
        # EMD parameters
        self.emd_config = config.get("emd", {})
        self.emd_features = self.emd_config.get("features", ["peak_amplitude", "std_df_f"])
        self.n_bins = self.emd_config.get("n_bins", 20)
    
    def run_analysis(self, fluorescence_data, roi_masks, metrics_df, output_dir):
        """
        Run all enabled advanced analyses.
        
        Parameters
        ----------
        fluorescence_data : numpy.ndarray
            Background-corrected fluorescence traces with shape (n_rois, n_frames)
        roi_masks : list
            List of ROI masks
        metrics_df : pandas.DataFrame
            DataFrame containing metrics for each ROI
        output_dir : str
            Directory to save the analysis results
            
        Returns
        -------
        dict
            Dictionary of analysis results
        """
        start_time = time.time()
        self.logger.info("Starting advanced fluorescence analysis")
        
        # Create output directory for advanced analysis
        adv_dir = os.path.join(output_dir, "advanced_analysis")
        os.makedirs(adv_dir, exist_ok=True)
        
        results = {}
        
        # Run machine learning classification if enabled
        if self.ml_enabled:
            ml_results = self.classify_roi_patterns(fluorescence_data, metrics_df, adv_dir)
            results["ml"] = ml_results
        
        # Run correlation analysis if enabled
        if self.correlation_enabled:
            corr_results = self.analyze_roi_correlations(fluorescence_data, roi_masks, adv_dir)
            results["correlation"] = corr_results
        
        # Run EMD analysis if enabled
        if self.emd_enabled and "pain_model" in metrics_df.columns:
            emd_results = self.compute_earth_movers_distance(metrics_df, adv_dir)
            results["emd"] = emd_results
        
        self.logger.info(f"Advanced analysis completed in {time.time() - start_time:.2f} seconds")
        
        # Save summary of results
        self.save_summary(results, adv_dir)
        
        return results
    
    def classify_roi_patterns(self, fluorescence_data, metrics_df, output_dir):
        """
        Apply machine learning to classify ROI activity patterns.
        
        Parameters
        ----------
        fluorescence_data : numpy.ndarray
            Background-corrected fluorescence traces with shape (n_rois, n_frames)
        metrics_df : pandas.DataFrame
            DataFrame containing metrics for each ROI
        output_dir : str
            Directory to save the classification results
            
        Returns
        -------
        dict
            Classification results
        """
        ml_start = time.time()
        self.logger.info("Starting ROI pattern classification")
        
        # Create output directory for ML results
        ml_dir = os.path.join(output_dir, "ml_classification")
        os.makedirs(ml_dir, exist_ok=True)
        
        # Extract features for classification
        features = metrics_df[self.feature_selection].copy()
        
        # Check for NaN values and handle them
        if features.isna().any().any():
            self.logger.warning("NaN values found in features, filling with mean values")
            features = features.fillna(features.mean())
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Check if we can use LDA (requires pain_model labels)
        use_lda = 'pain_model' in metrics_df.columns and len(metrics_df['pain_model'].unique()) > 1
        self.logger.info(f"Classification method: {'LDA' if use_lda else 'PCA+KMeans'}")
        
        # If we have pain model information, use LDA for supervised classification
        if use_lda:
            # Linear Discriminant Analysis for supervised classification
            pain_models = metrics_df['pain_model'].unique()
            n_components = min(len(pain_models) - 1, 3)  # LDA is limited to C-1 components
            
            # Create LDA model and fit
            lda = LDA(n_components=n_components)
            
            # Convert pain model strings to numeric labels for LDA
            pain_model_labels = pd.Categorical(metrics_df['pain_model']).codes
            
            # Fit LDA model
            features_lda = lda.fit_transform(features_scaled, pain_model_labels)
            
            # Store pain model predictions
            metrics_df['pain_model_predicted'] = lda.predict(features_scaled)
            metrics_df['pain_model_predicted'] = [pain_models[i] for i in metrics_df['pain_model_predicted']]
            
            # Get LDA explained variance (if available)
            if hasattr(lda, 'explained_variance_ratio_'):
                explained_variance = lda.explained_variance_ratio_
            else:
                # Calculate manual approximation of explained variance
                total_variance = np.var(features_scaled, axis=0).sum()
                transformed_variance = np.var(features_lda, axis=0)
                explained_variance = transformed_variance / total_variance
            
            # Plot LDA results
            plt.figure(figsize=(12, 10))
            
            # Use different plot types based on number of components
            if n_components == 1:
                # For 1 component, create a 1D strip plot
                plt.figure(figsize=(12, 6))
                for i, model in enumerate(pain_models):
                    model_data = features_lda[metrics_df['pain_model'] == model]
                    sns.kdeplot(model_data[:, 0], label=model, fill=True, alpha=0.5)
                
                plt.title('LDA Separation of ROIs by Pain Model')
                plt.xlabel('LDA Component 1')
                plt.ylabel('Density')
                
            elif n_components == 2:
                # For 2 components, create a 2D scatter plot
                plt.figure(figsize=(12, 10))
                for i, model in enumerate(pain_models):
                    mask = metrics_df['pain_model'] == model
                    plt.scatter(
                        features_lda[mask, 0], 
                        features_lda[mask, 1],
                        label=model,
                        alpha=0.7,
                        s=80
                    )
                
                # Add decision boundaries if possible
                try:
                    # Create a mesh grid to plot decision boundaries
                    x_min, x_max = features_lda[:, 0].min() - 1, features_lda[:, 0].max() + 1
                    y_min, y_max = features_lda[:, 1].min() - 1, features_lda[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                        np.arange(y_min, y_max, 0.1))
                    
                    # Transform mesh points back to original feature space
                    mesh_points = np.c_[xx.ravel(), yy.ravel()]
                    
                    # This part requires inverse_transform which LDA might not have
                    # Skipping for now but could be implemented with a custom approach
                except Exception as e:
                    self.logger.warning(f"Could not plot decision boundaries: {str(e)}")
                
                plt.xlabel(f'LDA Component 1 ({explained_variance[0]:.2%})')
                plt.ylabel(f'LDA Component 2 ({explained_variance[1]:.2%})')
                plt.title('LDA Separation of ROIs by Pain Model')
                
            elif n_components == 3:
                # For 3 components, create a 3D scatter plot
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                for i, model in enumerate(pain_models):
                    mask = metrics_df['pain_model'] == model
                    ax.scatter(
                        features_lda[mask, 0], 
                        features_lda[mask, 1],
                        features_lda[mask, 2],
                        label=model,
                        alpha=0.7,
                        s=80
                    )
                
                ax.set_xlabel(f'LDA Component 1 ({explained_variance[0]:.2%})')
                ax.set_ylabel(f'LDA Component 2 ({explained_variance[1]:.2%})')
                ax.set_zlabel(f'LDA Component 3 ({explained_variance[2]:.2%})')
                ax.set_title('LDA Separation of ROIs by Pain Model')
            
            plt.legend(title='Pain Model')
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, 'lda_pain_model_separation.png'), dpi=300)
            plt.close()
            
            # Get feature importance (LDA coefficients)
            if hasattr(lda, 'coef_'):
                coefficients = lda.coef_
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                
                # For each LDA component
                for i in range(min(3, coefficients.shape[0])):
                    # Sort coefficients
                    sorted_idx = np.argsort(np.abs(coefficients[i]))
                    pos = np.arange(sorted_idx.shape[0])
                    
                    plt.subplot(min(3, coefficients.shape[0]), 1, i+1)
                    plt.barh(pos, coefficients[i, sorted_idx], align='center')
                    plt.yticks(pos, np.array(self.feature_selection)[sorted_idx])
                    plt.title(f'LDA Component {i+1} Feature Importance')
                
                plt.tight_layout()
                plt.savefig(os.path.join(ml_dir, 'lda_feature_importance.png'), dpi=300)
                plt.close()
            
            # Calculate classification metrics
            accuracy = np.mean(metrics_df['pain_model'] == metrics_df['pain_model_predicted'])
            
            # Create confusion matrix
            cm = confusion_matrix(
                pd.Categorical(metrics_df['pain_model']).codes,
                pd.Categorical(metrics_df['pain_model_predicted']).codes
            )
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=pain_models,
                yticklabels=pain_models
            )
            plt.xlabel('Predicted Pain Model')
            plt.ylabel('True Pain Model')
            plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, 'lda_confusion_matrix.png'), dpi=300)
            plt.close()
            
            # Save LDA results for each ROI
            metrics_df.to_csv(os.path.join(ml_dir, 'roi_lda_results.csv'), index=False)
            
            # Analyze pain model traits
            model_stats = metrics_df.groupby('pain_model')[self.feature_selection].agg(['mean', 'std'])
            model_stats.to_csv(os.path.join(ml_dir, 'pain_model_statistics.csv'))
            
            # Return LDA results
            lda_results = {
                'method': 'LDA',
                'pain_models': pain_models.tolist(),
                'n_components': n_components,
                'explained_variance': explained_variance.tolist() if isinstance(explained_variance, np.ndarray) else None,
                'accuracy': accuracy,
                'features_used': self.feature_selection
            }
            
            # If slice type is also available, analyze pain model vs slice type relationship
            if 'slice_type' in metrics_df.columns:
                # Create contingency table
                contingency = pd.crosstab(metrics_df['pain_model'], metrics_df['slice_type'])
                
                # Chi-square test
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                
                contingency.to_csv(os.path.join(ml_dir, 'pain_model_by_slice_type.csv'))
                
                # Visualize relationship
                plt.figure(figsize=(10, 6))
                contingency.plot(kind='bar', stacked=True)
                plt.title(f'Pain Model Distribution by Slice Type (p={p:.4f})')
                plt.xlabel('Pain Model')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(ml_dir, 'pain_model_by_slice_type.png'), dpi=300)
                plt.close()
        
        else:
            # If no pain model information, fall back to unsupervised clustering
            self.logger.info("No pain model information found, using PCA and k-means clustering")
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=min(3, len(self.feature_selection)))
            features_pca = pca.fit_transform(features_scaled)
            
            # Unsupervised clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Add cluster assignments to metrics DataFrame
            metrics_df['cluster'] = clusters
            
            # Visualize clusters in PCA space
            plt.figure(figsize=(10, 8))
            for i in range(self.n_clusters):
                mask = clusters == i
                plt.scatter(
                    features_pca[mask, 0], 
                    features_pca[mask, 1],
                    label=f'Cluster {i+1}',
                    alpha=0.7
                )
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.title('ROI Clusters in PCA Space')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, 'roi_clusters_pca.png'), dpi=300)
            plt.close()
            
            # Analyze cluster characteristics
            cluster_stats = metrics_df.groupby('cluster')[self.feature_selection].agg(['mean', 'std'])
            
            # Save results
            metrics_df.to_csv(os.path.join(ml_dir, 'roi_with_clusters.csv'), index=False)
            cluster_stats.to_csv(os.path.join(ml_dir, 'cluster_statistics.csv'))
            
            # If slice type is available, analyze relationship between clusters and slice type
            if 'slice_type' in metrics_df.columns:
                # Create contingency table
                contingency = pd.crosstab(metrics_df['cluster'], metrics_df['slice_type'])
                
                # Chi-square test
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                
                contingency.to_csv(os.path.join(ml_dir, 'cluster_by_slice_type.csv'))
                
                # Visualize relationship
                plt.figure(figsize=(10, 6))
                contingency.plot(kind='bar', stacked=True)
                plt.title(f'Cluster Distribution by Slice Type (p={p:.4f})')
                plt.xlabel('Cluster')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(ml_dir, 'cluster_by_slice_type.png'), dpi=300)
                plt.close()
            
            # Time series pattern analysis
            # Extract sample traces from each cluster
            n_samples = min(3, min(np.bincount(clusters)))
            sample_traces = {}
            
            for i in range(self.n_clusters):
                cluster_indices = np.where(clusters == i)[0]
                sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)
                sample_traces[i] = fluorescence_data[sampled_indices]
            
            # Plot sample traces from each cluster
            plt.figure(figsize=(12, 3 * self.n_clusters))
            for i in range(self.n_clusters):
                plt.subplot(self.n_clusters, 1, i+1)
                for j in range(n_samples):
                    plt.plot(sample_traces[i][j], alpha=0.7)
                plt.title(f'Cluster {i+1} Sample Traces')
                plt.xlabel('Frame')
                plt.ylabel('dF/F')
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, 'cluster_sample_traces.png'), dpi=300)
            plt.close()
            
            # Calculate average trace for each cluster
            avg_traces = np.zeros((self.n_clusters, fluorescence_data.shape[1]))
            for i in range(self.n_clusters):
                mask = clusters == i
                if np.any(mask):
                    avg_traces[i] = np.mean(fluorescence_data[mask], axis=0)
            
            # Plot average traces
            plt.figure(figsize=(12, 6))
            for i in range(self.n_clusters):
                plt.plot(avg_traces[i], label=f'Cluster {i+1}')
            plt.title('Average Fluorescence Traces by Cluster')
            plt.xlabel('Frame')
            plt.ylabel('dF/F')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, 'cluster_average_traces.png'), dpi=300)
            plt.close()
            
            # Return unsupervised clustering results
            lda_results = {
                'method': 'PCA+KMeans',
                'n_clusters': self.n_clusters,
                'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
                'cluster_counts': np.bincount(clusters).tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'features_used': self.feature_selection
            }
        
        # If pain model info is available, create comparison of fluorescence traces by pain model
        if 'pain_model' in metrics_df.columns:
            # Average trace for each pain model
            pain_models = metrics_df['pain_model'].unique()
            avg_model_traces = {}
            
            for model in pain_models:
                model_indices = np.where(metrics_df['pain_model'] == model)[0]
                if len(model_indices) > 0:
                    avg_model_traces[model] = np.mean(fluorescence_data[model_indices], axis=0)
            
            # Plot average traces by pain model
            plt.figure(figsize=(12, 6))
            for model, trace in avg_model_traces.items():
                plt.plot(trace, label=model, linewidth=2)
            plt.title('Average Fluorescence Traces by Pain Model')
            plt.xlabel('Frame')
            plt.ylabel('dF/F')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, 'pain_model_average_traces.png'), dpi=300)
            plt.close()
        
        self.logger.info(f"ROI pattern classification completed in {time.time() - ml_start:.2f} seconds")
        
        # Return results
        return lda_results
    
    def analyze_roi_correlations(self, fluorescence_data, roi_masks, output_dir):
        """
        Analyze correlations between nearby ROIs.
        
        Parameters
        ----------
        fluorescence_data : numpy.ndarray
            Background-corrected fluorescence traces with shape (n_rois, n_frames)
        roi_masks : list
            List of ROI masks
        output_dir : str
            Directory to save the correlation results
            
        Returns
        -------
        dict
            Correlation analysis results
        """
        corr_start = time.time()
        self.logger.info("Starting ROI correlation analysis")
        
        # Create output directory for correlation results
        corr_dir = os.path.join(output_dir, "roi_correlations")
        os.makedirs(corr_dir, exist_ok=True)
        
        n_rois = len(roi_masks)
        
        # Calculate ROI centroids
        centroids = []
        for mask in roi_masks:
            y_indices, x_indices = np.where(mask)
            centroid_y = np.mean(y_indices)
            centroid_x = np.mean(x_indices)
            centroids.append((centroid_y, centroid_x))
        
        centroids = np.array(centroids)
        
        # Calculate pairwise distances between ROIs
        distances = squareform(pdist(centroids, 'euclidean'))
        
        # Calculate correlation matrix
        if self.corr_method == 'pearson':
            corr_matrix = np.corrcoef(fluorescence_data)
        elif self.corr_method == 'spearman':
            corr_matrix = np.zeros((n_rois, n_rois))
            for i in range(n_rois):
                for j in range(n_rois):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr_matrix[i, j], _ = stats.spearmanr(fluorescence_data[i], fluorescence_data[j])
        else:
            self.logger.warning(f"Unknown correlation method: {self.corr_method}, using pearson")
            corr_matrix = np.corrcoef(fluorescence_data)
        
        # Create distance-filtered correlation matrix
        nearby_corr_matrix = corr_matrix.copy()
        nearby_corr_matrix[distances > self.distance_threshold] = np.nan
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        mask = np.eye(n_rois, dtype=bool)  # Mask the diagonal
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True
        )
        plt.title(f'ROI Activity Correlation Matrix ({self.corr_method})')
        plt.tight_layout()
        plt.savefig(os.path.join(corr_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
        
        # Plot correlation vs distance scatter
        plt.figure(figsize=(10, 6))
        dist_flat = distances.flatten()
        corr_flat = corr_matrix.flatten()
        mask = ~np.eye(n_rois, dtype=bool).flatten()  # Remove diagonals
        
        plt.scatter(dist_flat[mask], corr_flat[mask], alpha=0.3)
        plt.title('Correlation vs Distance Between ROIs')
        plt.xlabel('Distance (pixels)')
        plt.ylabel(f'Correlation ({self.corr_method})')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=self.distance_threshold, color='g', linestyle='--', alpha=0.5, 
                   label=f'Distance Threshold: {self.distance_threshold}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(corr_dir, 'correlation_vs_distance.png'), dpi=300)
        plt.close()
        
        # Identify highly correlated ROI pairs
        high_corr_threshold = 0.7
        high_corr_pairs = []
        
        for i in range(n_rois):
            for j in range(i+1, n_rois):
                if (distances[i, j] <= self.distance_threshold and 
                    corr_matrix[i, j] >= high_corr_threshold):
                    high_corr_pairs.append({
                        'roi1': i+1,  # 1-indexed for user-friendliness
                        'roi2': j+1,  # 1-indexed for user-friendliness
                        'correlation': corr_matrix[i, j],
                        'distance': distances[i, j]
                    })
        
        # Create a graph visualization of correlated ROIs
        if len(high_corr_pairs) > 0:
            try:
                import networkx as nx
                
                G = nx.Graph()
                
                # Add nodes (ROIs)
                for i in range(n_rois):
                    G.add_node(i, pos=(centroids[i][1], centroids[i][0]))  # x, y for visualization
                
                # Add edges (correlations)
                for pair in high_corr_pairs:
                    G.add_edge(
                        pair['roi1']-1,  # Convert back to 0-indexed
                        pair['roi2']-1,  # Convert back to 0-indexed
                        weight=pair['correlation']
                    )
                
                # Get positions
                pos = nx.get_node_attributes(G, 'pos')
                
                # Plot the graph
                plt.figure(figsize=(12, 10))
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')
                
                # Draw edges with color based on correlation strength
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, 
                                      edge_color=weights, edge_cmap=plt.cm.coolwarm)
                
                # Add labels
                nx.draw_networkx_labels(G, pos, font_size=8)
                
                plt.title('Correlated ROI Network')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(corr_dir, 'correlated_roi_network.png'), dpi=300)
                plt.close()
                
            except ImportError:
                self.logger.warning("NetworkX not installed, skipping correlation network visualization")
        
        # Save correlation data
        np.save(os.path.join(corr_dir, 'correlation_matrix.npy'), corr_matrix)
        np.save(os.path.join(corr_dir, 'roi_distances.npy'), distances)
        
        if high_corr_pairs:
            pd.DataFrame(high_corr_pairs).to_csv(
                os.path.join(corr_dir, 'high_correlation_pairs.csv'), 
                index=False
            )
        
        self.logger.info(f"ROI correlation analysis completed in {time.time() - corr_start:.2f} seconds")
        
        return {
            'method': self.corr_method,
            'distance_threshold': self.distance_threshold,
            'high_correlation_threshold': high_corr_threshold,
            'high_correlation_pairs_count': len(high_corr_pairs),
            'average_correlation': np.nanmean(nearby_corr_matrix)
        }
    
    def compute_earth_movers_distance(self, metrics_df, output_dir):
        """
        Compute Earth Mover's Distance between pain models.
        
        Parameters
        ----------
        metrics_df : pandas.DataFrame
            DataFrame containing metrics for each ROI
        output_dir : str
            Directory to save the EMD results
            
        Returns
        -------
        dict
            EMD results
        """
        emd_start = time.time()
        self.logger.info("Starting Earth Mover's Distance analysis for pain models")
        
        # Create output directory for EMD results
        emd_dir = os.path.join(output_dir, "emd_analysis")
        os.makedirs(emd_dir, exist_ok=True)
        
        # Check if pain model data is available
        if "pain_model" not in metrics_df.columns:
            self.logger.warning("No pain model data found, skipping EMD analysis")
            return {
                'error': 'No pain model data found'
            }
        
        # Get unique pain models
        pain_models = metrics_df["pain_model"].unique()
        
        if len(pain_models) < 2:
            self.logger.warning(f"Only one pain model found ({pain_models[0]}), EMD analysis requires at least two")
            return {
                'error': 'Only one pain model found'
            }
        
        # Prepare feature data
        emd_results = {}
        
        for feature in self.emd_features:
            if feature not in metrics_df.columns:
                self.logger.warning(f"Feature {feature} not found in metrics, skipping")
                continue
            
            # Create histograms for each pain model
            histograms = {}
            bin_edges = None
            
            # Find global min and max for consistent binning
            feature_min = metrics_df[feature].min()
            feature_max = metrics_df[feature].max()
            
            # Add a small buffer to prevent edge issues
            buffer = (feature_max - feature_min) * 0.05
            feature_min -= buffer
            feature_max += buffer
            
            for model in pain_models:
                model_data = metrics_df[metrics_df["pain_model"] == model][feature].dropna()
                
                if len(model_data) == 0:
                    self.logger.warning(f"No data for pain model {model}, skipping")
                    continue
                
                # Create histogram
                hist, bin_edges = np.histogram(
                    model_data, 
                    bins=self.n_bins, 
                    range=(feature_min, feature_max),
                    density=True
                )
                
                histograms[model] = hist
            
            # Compute EMD between all pain model pairs
            model_pairs = []
            emd_values = []
            
            for i, model1 in enumerate(pain_models):
                if model1 not in histograms:
                    continue
                    
                for model2 in pain_models[i+1:]:
                    if model2 not in histograms:
                        continue
                    
                    # Compute EMD
                    # Normalize histograms
                    hist1 = histograms[model1] / np.sum(histograms[model1])
                    hist2 = histograms[model2] / np.sum(histograms[model2])
                    
                    # Create distance matrix (1D case: absolute difference in bin positions)
                    n_bins = len(hist1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Compute distance matrix between bins
                    dist_matrix = np.abs(bin_centers.reshape(-1, 1) - bin_centers.reshape(1, -1))
                    
                    # Normalize distance matrix
                    dist_matrix /= dist_matrix.max()
                    
                    try:
                        # Compute EMD
                        emd_value = ot.emd2(hist1, hist2, dist_matrix)
                        
                        model_pairs.append(f"{model1} vs {model2}")
                        emd_values.append(emd_value)
                        
                    except Exception as e:
                        self.logger.error(f"Error computing EMD for {model1} vs {model2}: {str(e)}")
                        continue
            
            # Visualize histograms
            plt.figure(figsize=(12, 6))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            for model, hist in histograms.items():
                plt.plot(bin_centers, hist, label=model, linewidth=2, alpha=0.7)
            
            plt.title(f'Distribution of {feature} Across Pain Models')
            plt.xlabel(feature)
            plt.ylabel('Probability Density')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(emd_dir, f'{feature}_distribution.png'), dpi=300)
            plt.close()
            
            # Visualize EMD results
            if model_pairs:
                plt.figure(figsize=(10, 6))
                plt.bar(model_pairs, emd_values)
                plt.title(f'Earth Mover\'s Distance for {feature} Between Pain Models')
                plt.xlabel('Pain Model Pair')
                plt.ylabel('EMD')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(emd_dir, f'{feature}_emd.png'), dpi=300)
                plt.close()
            
            # Save EMD results for this feature
            emd_results[feature] = {
                'model_pairs': model_pairs,
                'emd_values': emd_values
            }
        
        # Save overall EMD results
        if emd_results:
            with open(os.path.join(emd_dir, 'emd_summary.txt'), 'w') as f:
                f.write("Earth Mover's Distance Analysis Summary\n")
                f.write("=====================================\n\n")
                
                for feature, results in emd_results.items():
                    f.write(f"Feature: {feature}\n")
                    f.write("-----------------\n")
                    
                    if results['model_pairs']:
                        for i, pair in enumerate(results['model_pairs']):
                            f.write(f"{pair}: {results['emd_values'][i]:.4f}\n")
                    else:
                        f.write("No EMD results for this feature\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Earth Mover's Distance analysis completed in {time.time() - emd_start:.2f} seconds")
        
        return emd_results
    
    def save_summary(self, results, output_dir):
        """
        Save a summary of all advanced analysis results.
        
        Parameters
        ----------
        results : dict
            Dictionary of analysis results
        output_dir : str
            Directory to save the summary
        """
        summary_path = os.path.join(output_dir, "advanced_analysis_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Advanced Fluorescence Analysis Summary\n")
            f.write("=====================================\n\n")
            
            # ML Results
            if 'ml' in results:
                f.write("Machine Learning Classification\n")
                f.write("-----------------------------\n")
                ml = results['ml']
                
                if ml['method'] == 'LDA':
                    f.write(f"Method: Linear Discriminant Analysis (LDA)\n")
                    f.write(f"Pain models: {ml['pain_models']}\n")
                    f.write(f"Number of LDA components: {ml['n_components']}\n")
                    if ml.get('explained_variance'):
                        f.write(f"Explained variance: {[f'{v:.2%}' for v in ml['explained_variance']]}\n")
                    f.write(f"Classification accuracy: {ml['accuracy']:.2%}\n")
                    f.write(f"Features used: {ml['features_used']}\n\n")
                else:
                    f.write(f"Method: {ml['method']}\n")
                    f.write(f"Number of clusters: {ml['n_clusters']}\n")
                    f.write(f"Cluster sizes: {ml['cluster_counts']}\n")
                    if 'pca_variance_explained' in ml:
                        f.write(f"PCA variance explained: {[f'{v:.2%}' for v in ml['pca_variance_explained']]}\n")
                    f.write(f"Features used: {ml['features_used']}\n\n")
            
            # Correlation Results
            if 'correlation' in results:
                f.write("ROI Correlation Analysis\n")
                f.write("-----------------------\n")
                corr = results['correlation']
                
                f.write(f"Correlation method: {corr['method']}\n")
                f.write(f"Distance threshold: {corr['distance_threshold']} pixels\n")
                f.write(f"High correlation threshold: {corr['high_correlation_threshold']}\n")
                f.write(f"Number of highly correlated pairs: {corr['high_correlation_pairs_count']}\n")
                f.write(f"Average correlation between nearby ROIs: {corr['average_correlation']:.4f}\n\n")
            
            # EMD Results
            if 'emd' in results:
                f.write("Earth Mover's Distance Analysis\n")
                f.write("----------------------------\n")
                
                if 'error' in results['emd']:
                    f.write(f"Error: {results['emd']['error']}\n\n")
                else:
                    for feature, feature_results in results['emd'].items():
                        f.write(f"Feature: {feature}\n")
                        
                        if feature_results['model_pairs']:
                            for i, pair in enumerate(feature_results['model_pairs']):
                                f.write(f"  {pair}: {feature_results['emd_values'][i]:.4f}\n")
                        else:
                            f.write("  No EMD results for this feature\n")
                        
                        f.write("\n")
        
        self.logger.info(f"Saved advanced analysis summary to {summary_path}")