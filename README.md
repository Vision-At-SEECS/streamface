# StreamFace

## Abstract

Recent advancements in deep learning have significantly improved face recognition system performance, but automating recognition in large-scale video content remains a formidable challenge. Current video face recognition approaches are primarily tailored for short-form content, such as movies and television shows, which feature a limited number of faces and identities. Moreover, most large-scale face datasets are derived from web images and do not effectively capture the complexities of the video domain. In response to these limitations, we present the StreamFace framework as the cornerstone of our work. StreamFace represents a pioneering approach to annotation, leveraging clustering-based, semi-automatic techniques specially designed for efficiently annotating vast collections of face images. It is this innovative framework that has empowered us to curate the TVFace dataset, a remarkable resource in its own right. TVFace comprises a staggering 2.6 million face images spanning 33 thousand distinct identities, all extracted from public live streams of international news channels. TVFace, underpinned by the StreamFace framework, establishes itself as the first and foremost large-scale face dataset sourced exclusively from long-form video content. It serves as a critical tool for evaluating and advancing face representation and identity classification components across both image and video domains, thereby holding immense value for researchers and practitioners alike. Furthermore, the adaptability of TVFace extends beyond traditional recognition tasks, encompassing identification and clustering. To demonstrate the practicality of TVFace, we introduce a tree search-based Hierarchical Retrieval Index tailored for rapid face matching. This showcase underscores the effectiveness of TVFace in evaluating real-time person retrieval systems. In conclusion, our work centers around the StreamFace framework's pioneering role in enabling the creation of TVFace, a dataset poised to reshape the landscape of face recognition in the video domain, making it a crucial resource for the research community.

![Alt text](figures/dataset_overview.png)

Please fill the form below to send a request for acquiring the download link for the dataset from the authors.

https://docs.google.com/forms/d/165AYNU9iGQA-wEpH68jEfjnGzJdF_ljFkZNSjOaO3qU

## Streamface 

A framework for extraction of face datasets from YouTube videos and livestreams.

![Alt text](figures/method.png)

### Installation

Clone repository and install required packages using
```
pip install -r requirements.txt
```

### Usage

The StreamFace framework consists of multiple steps that are carried out in order to curate a dataset of faces from a long form video content such as live streams.

#### 1. Frame Collection

When the user executes the code provided below, the following operations will be performed:

1. Video compression-based keyframe extraction will be utilized to select keyframes from the video, ensuring sparse sampling of high-quality images with minimal noise and motion blur. These keyframes will adapt to scene changes.

2. Content-based frame analysis will be employed to identify and exclude noisy, empty, and duplicate frames from the selection.

3. The selected frames will be saved to disk, and their respective timestamps will be used as filenames for storage.

```python
from streamface.stream_capture import StreamCapture

capture = StreamCapture(
    name='skynews',
    url='https://www.youtube.com/watch?v=9Auq9mYxFEE',
    output_dir='./data/skynews',
    batch_size=50,
    empty_threshold=0.95,
    blur_threshold=50,
    similarity_threshold=0.9,
    reconnect_interval=500,
    log_interval=100
)

capture.call()
```

#### 2. Face Extraction


The following code utilizes RetinaFace and it performs the following operations:

1. Utilizes RetinaFace for face detection and extraction from stored frames.

2. Implements minimum confidence and size thresholds for the detected faces.

3. Enlarges the predicted bounding boxes to encompass the surrounding region of the faces.

4. Crops out the enlarged regions containing the faces and resizes them to 256×256 pixels.

5. Stores the extracted face images to disk, using filenames derived from the filenames of the corresponding frames. This preserves the temporal order of faces based on the timestamps of their on-screen appearances.

```python
from streamface.face_extraction import FaceExtraction

extract = FaceExtraction(
    input_dir='./data/skynews',
    output_dir='./data/skynews',
    detection='retinaface',
    batch_size=32,
    conf_threshold=0.95,
    size_threshold=0.005,
    blur_threshold=25,
    match_thresholds=(0.75, 0.75),
    face_size=(256, 256),
    aligned_size=(128, 128),
    padding=0.5,
    margin=1.0,
    resume=True,
    log_interval=100
)

extract.call()
```

#### 3. Facial Feature Extraction Prior to Clustering 

Upon executing the code, the following operations are carried out:

1. Deployment of an ensemble consisting of two feature representation models for generating feature vectors.

2. The first model utilizes a ResNet34 backbone, which was trained on the CASIA-WebFace dataset using the ArcFace loss function. This model produces 512-dimensional feature vectors.

3. The second model employs an InceptionResnetV2 backbone and was trained on the VGGFace2 dataset using softmax loss, also yielding 512-dimensional feature vectors.

4. Both models are imported pretrained from the DeepFace library.

5. The outputs from each individual model are L2 normalized.

6. The normalized outputs are then combined using summation, followed by another round of normalization.

7. Additionally, face alignment is performed by closely cropping the face and adjusting the image rotation until the eyes are horizontally aligned.

8. The resulting feature set is stored in the hard disk in .pkl format

```python
from streamface.face_analysis import FaceAnalysis

analyze = FaceAnalysis(
    input_dir='./data/skynews',
    output_dir='./data/skynews',
    representation='arcfacenet',
    demographics='fairface',
    expression='dfemotion',
    mask='chandrikanet',
    pose='whenet',
    batch_size=128,
    resume=True,
    log_interval=100
)

analyze.call()
```

#### 4. Feature Evaluation

An additional evaluation step is carried out for determining the matching threshould for the clustering step. The code utilizes intrinsic evaluation techniques, specifically the Silhouette Coefficient, as described in the reference. This evaluation method is employed alongside manual verification to establish the ideal distance threshold for clustering. After analysis, it was determined that the optimal cosine distance threshold falls within the range of 0.2 to 0.3 for various channels or contexts.

```python
from streamface.feature_evaluation import FeatureEvaluation


model = FeatureEvaluation(
    name='skynews',
    features_path='./data/skynews/features/xyz.pkl',
    metric='cosine',
    max_samples=30000,
    k=5000,
    thresholds=list(np.arange(0.1, 0.4, 0.025))
)

model.evaluate()
```

#### 5. Face Clustering

When the code is executed, it utilizes the agglomerative clustering method with complete linkage, as outlined in the work by Defays (1977) \cite{Defays1977}. This specific clustering algorithm is chosen because it relies on the pairwise distances between feature vectors, which are directly optimized by the feature representation models.

The complete linkage criteria, known for producing closely-knit and "spherical" clusters, are employed. This choice of criteria leads to the formation of clusters with high purity. In this clustering process, the cosine distance metric is used to measure the similarity between feature vectors.

The matching threshold resulting from the previous step is 0.3.

```python
from streamface.face_clustering import FaceClustering


cluster = FaceClustering(
    features_path='./data/skynews/features/xyz.pkl',
    matches_path='./data/skynews/metadata/matches.csv',
    faces_dir='./data/skynews/faces',
    output_dir='./data/skynews',
    metric='cosine',
    linkage='complete',
    matching_threshold=0.3,
    matching_batchsize=20000,
    merging_threshold=0.5,
)

cluster.call()
```

#### 6. Cluster Evaluation

An optional step for evaluting the resulting clusters from the previous steps.

```python
from streamface.cluster_evaluation import ClusterEvaluation


model = ClusterEvaluation(
    name='skynews',
    features_path='./data/skynews/features/xyz.pkl',
    annotations_path='./data/skynews/metadata/annotations.json',
    scores_path='./data/skynews/metadata/cluster_scores.pkl',
    plots_path='./data/skynews/metadata',
    metric='cosine',
)

model.evaluate()
```

#### 7. Cluster Matching

In the clustering process, clusters are assigned to the same class if the distance between their mean vectors is less than a specified minimum threshold (min_threshold). Cluster pairs with mean vector distances greater than 0.5 are disregarded and not used for annotation. However, cluster pairs with distances falling within the range of 0.35 to 0.5 are specifically chosen for annotation purposes.

```python
from streamface.cluster_matching import ClusterMatching


model = ClusterMatching(
    features_path='./data/skynews/features/xyz.pkl',
    annotations_path='./data/skynews/metadata/annotations.json',
    evals_path='./data/skynews/metadata/cluster_evals.json',
    metric='cosine',
    topk=10000,
    min_threshold=0.35,
    max_threshold=0.5,
    mode='average',
)

model.match()
```

#### 8. Cluster Refinement

Clustering results in two kinds of erroneous groupings that need fixing: multiple clusters for the same identity and multiple identities in one cluster. The former were corrected by merging all clusters for the same identity while the latter were excluded from the dataset.

Cluster refinement was performed on cluster pairs so that the annotator could examine them side by side to determine whether they contained images of the same person, while also checking both clusters for noise. For all cluster pairs, at most K face images were sampled from both clusters and shown to the human annotator. If all faces belonged to the same person, the clusters were marked for merging. If either of the clusters contained multiple identities or non-face images, it was marked as noisy and excluded from the dataset. Since the number of negative cluster pairs is far greater than positive pairs, the search space was limited to a list of most similar clusters. The mean feature vector from each cluster was used to compute pairwise cosine similarity between all clusters. A lenient threshold was then applied to exclude all pairs that were clearly dissimilar and the remaining pairs, sorted by similarity score, were selected for manual examination.

```python
from streamface.fiftyone_annotation import FiftyOneAnnotation


model = FiftyOneAnnotation(
    name='skynews',
    faces_path='./data/skynews/faces',
    evals_path='./data/skynews/metadata/cluster_evals.json',
    annotations_path='./data/skynews/metadata/annotations.json',
    new_annotations_path='./data/skynews/metadata/annotations_manual.json',
)

model.annotate()
```

#### 9. Evaluate Annotations

This step compares the labels obtained after the "Cluster Refinement" step with the labels before the refinement to evaluate the quality of automatic clustering.

```python
from streamface.annotation_evaluation import AnnotationEvaluation


evaluator = AnnotationEvaluation(
    true_annotations_path='./data/skynews/metadata/annotations_manual.json',
    pred_annotations_path='./data/skynews/metadata/annotations.json',
)

evaluator.evaluate(verbose=True)
```

## Hierarchical Retrieval Index 

Tree search-based fast face matching algorithm.

[Link to Repository](/hri)
