# Install
Requirements:  
[Python](https://www.python.org/downloads/)  
[nodeJs](https://nodejs.org/en/download/)

To install:
```npm install;pip install -r requirements.txt```

To run: 
`python run.py` to run the project fully or
`node sequential.js` to just generate images.

Configuration:
settings.json
`style`: the style of the generated art  
can be a singular value, or an array  
if it is an array, its length must be equal to `iterations`  
see styles.js for more details  
`file_folder`: the name of the folder the images will be generated in  
`prompt`: the prompt used for the art  
can be a singular value or an array  
if it is an array, its length must be equal to `iterations`  
`quiet`: whether to log out all details or not  
`inter`: download the intermediate images  
`inputImages`: a path, or list of paths leading to input images.  
must be in the `jpeg` file format  
`iterations`: how many times to generate art.  
