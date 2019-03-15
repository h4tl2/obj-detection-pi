## Real-time Object Detection on Movidius  
### Run Camerastream  
```
python ncs_realtime_objectdetection.py --graph [path to graph] --display 1
```  
### Run Videostream
```
python ncs_video_objectdetection.py --graph [path to graph] --display 1 --video [path to video]
```
### Flow  
1.Export your model's graphdef (you can use tf.train.write_graph to export the graphdef from within your tf session)  
2.Freeze your graphdef using freeze_graph.py (you'll need the checkpoint files)  
3.Pass the frozen model to mvNCCompile along with the input and output nodes.  
### Caffe  
```
mvNCCompile network.prototxt [-w network.caffemodel] [-s max_number_of_shaves] [-in input_node_name] [-on output_node_name] [-is input_width input_height] [-o output_graph_filename] [-ec]
```
### Tensorflow  
```
mvNCCompile network.meta [-s max_number_of_shaves] [-in input_node_name] [-on output_node_name] [-is input_width input_height] [-o output_graph_filename] [-ec]
```
### download large file  
credit to ![github](https://github.com/circulosmeos/gdown.pl)
```
./gdown.pl https://drive.google.com/file/d/0B1L_hFrWJfRhLUJZdXdSdTdfSWs/edit axolotl.mp4   
```
### graphs
```
mobilenetgraph https://drive.google.com/file/d/1mRT27wLAHswfOScc6CX74Yr5a1lbDqwO/view?usp=sharing 
```