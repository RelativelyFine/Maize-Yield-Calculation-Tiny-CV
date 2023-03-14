import React, { useState, useEffect } from "react";
import Image from "next/image";
import FilePreview from "./FilePreview";
import styles from "../styles/DropZone.module.css";
import { InfinitySpin } from "react-loader-spinner";

const DropZone = ({ data, dispatch }) => {
  // onDragEnter sets inDropZone to true
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dispatch({ type: "SET_IN_DROP_ZONE", inDropZone: true });
  };

  // onDragLeave sets inDropZone to false
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();

    dispatch({ type: "SET_IN_DROP_ZONE", inDropZone: false });
  };

  // onDragOver sets inDropZone to true
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();

    // set dropEffect to copy i.e copy of the source item
    e.dataTransfer.dropEffect = "copy";
    dispatch({ type: "SET_IN_DROP_ZONE", inDropZone: true });
  };

  // onDrop sets inDropZone to false and adds files to fileList
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();

    // get files from event on the dataTransfer object as an array
    let files = [...e.dataTransfer.files];

    // ensure a file or files are dropped
    if (files && files.length > 0) {
      // loop over existing files
      const existingFiles = data.fileList.map((f) => f.name);
      // check if file already exists, if so, don't add to fileList
      // this is to prevent duplicates
      files = files.filter((f) => !existingFiles.includes(f.name));

      // dispatch action to add droped file or files to fileList
      dispatch({ type: "ADD_FILE_TO_LIST", files });
      // reset inDropZone to false
      dispatch({ type: "SET_IN_DROP_ZONE", inDropZone: false });
    }
  };

  // handle file selection via input element
  const handleFileSelect = (e) => {
    // get files from event on the input element as an array
    let files = [...e.target.files];

    // ensure a file or files are selected
    if (files && files.length > 0) {
      // loop over existing files
      const existingFiles = data.fileList.map((f) => f.name);
      // check if file already exists, if so, don't add to fileList
      // this is to prevent duplicates
      files = files.filter((f) => !existingFiles.includes(f.name));

      // dispatch action to add selected file or files to fileList
      dispatch({ type: "ADD_FILE_TO_LIST", files });
    }
  };

  // store returned data from server in a variable
  const [numTassels, setNumTassels] = useState(null);

  // set loading to true while uploading files
  const [loading, setLoading] = useState(false);

  const [earsPer, setEarsPer] = useState(1);

  const [weight, setWeight] = useState(340);

  const [yieldWeight, setYieldWeight] = useState(0);

  const [yieldTassel, setYieldTassel] = useState(0);

  const [heatmapImages, setHeatmapImages] = useState([]);

  const [timeTaken, setTimeTaken] = useState(0);

  const [uploadTime, setUploadTime] = useState(0);

  const [heatmapTime, setHeatmapTime] = useState(0);

  // to handle file uploads
  const uploadFiles = async () => {
    // set loading to true
    setLoading(true);

    let start = new Date().getTime();
    // get the files from the fileList as an array
    let files = data.fileList;
    // initialize formData object
    const formData = new FormData();
    // loop over files and add to formData
    files.forEach((file) => formData.append("file", file));

    // Upload the files as a POST request to the server using fetch
    // Note: /api/fileupload is not a real endpoint, it is just an example
    const response = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData,
    });

    //successful file upload
    if (response.ok) {
      response
        .json()
        .then((data) => {
          console.log(data);
          // set the number of tassels returned from the server
          setNumTassels(data.total);
          setTimeTaken(data.time);
          setHeatmapImages(data.heatmaps);
          setHeatmapTime(data.heatmap_time * 1);
        })
        .catch((err) => {
          alert("Error uploading files");
        })
        .finally(() => {
          setLoading(false);
          let end = new Date().getTime();
          setUploadTime((end - start) / 1000 - timeTaken);
        });
    } else {
      // unsuccessful file upload
      setLoading(false);
      alert("Error uploading files");
    }
  };

  // calculate number of ears per plant
  useEffect(() => {
    if (numTassels) {
      let tempEarsPer = earsPer ? earsPer : 1;
      if (weight) {
        setYieldWeight((numTassels * weight * tempEarsPer).toFixed(2));
      } else {
        setYieldWeight((numTassels * 340 * tempEarsPer).toFixed(2));
      }
      setYieldTassel((numTassels * tempEarsPer).toFixed(2));
    }
  }, [numTassels, earsPer, weight]);

  return (
    <>
      <h3>&darr; Step 1: Upload (Multiple) Images &darr;</h3>
      <div
        className={styles.dropzone}
        onDrop={(e) => handleDrop(e)}
        onDragOver={(e) => handleDragOver(e)}
        onDragEnter={(e) => handleDragEnter(e)}
        onDragLeave={(e) => handleDragLeave(e)}
      >
        {/* color my svg /upload.svg white*/}

        <Image src="/upload.svg" width={50} height={50} />

        <input
          id="fileSelect"
          type="file"
          multiple
          className={styles.files}
          onChange={(e) => handleFileSelect(e)}
        />
        <label htmlFor="fileSelect">Upload Images</label>

        <h3 className={styles.uploadMessage}>
          or drag &amp; drop your images here
        </h3>
      </div>
      {/* Only show upload button after selecting atleast 1 file */}
      {data.fileList.length > 0 && (
        <>
          <h3>&darr; Step 2: Calculate Yield &darr;</h3>
          <button className={styles.uploadBtn} onClick={uploadFiles}>
            Calculate!
          </button>
        </>
      )}
      {loading && <InfinitySpin width="200" color="#fff" />}
      {numTassels && (
        <>
          <h3>&darr; Step 3: Input These Values! (optional) &darr;</h3>
          <div>
            <label htmlFor="earsPer">Average ears per plant: </label>
            <input
              id="earsPer"
              placeholder="Enter Value (default 1)"
              type="number"
              onChange={(e) => setEarsPer(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="weight">Average weight per plant (g): </label>
            <input
              id="weight"
              placeholder="Enter Value (default 340)"
              type="number"
              onChange={(e) => setWeight(e.target.value)}
            />
          </div>
          <br />
          {/* Only show the number of tassels after uploading files */}
          {!loading && numTassels && (
            <>
              <h3>&darr; Step 4: Results &darr;</h3>
              <div className={styles.results}>
                <div className={styles.resultBorder}>
                  Total Count (Ears): {yieldTassel} &plusmn;5% <br />
                  Total Count (Weight): {(yieldWeight / 1000).toFixed(2)} kg
                  &plusmn;5%
                </div>
                <br />
                Proccessing Time Taken: {(timeTaken * 1000).toFixed(2)}{" "}
                miliseconds
                <br />
                Upload Time Taken: {uploadTime.toFixed(2)} seconds
                <br />
                Heatmap Generation Time Taken: {heatmapTime.toFixed(2)} seconds
              </div>
              {/* Only show the heatmap images after uploading files */}
              {heatmapImages.length > 0 && (
                <>
                  <h3>&darr; Step 5: Heatmaps &darr;</h3>
                  <div className={styles.heatmapContainer}>
                    {heatmapImages.map((image, index) => {
                      image = "data:image/png;base64," + image;
                      return (
                        <div key={index} className={styles.heatmap}>
                          <Image src={image} width={754} height={500} />
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </>
          )}
          <br />
          {/* Pass the selectect or dropped files as props */}
        </>
      )}
      {data.fileList.length > 0 && <FilePreview fileData={data} />}
    </>
  );
};

export default DropZone;
