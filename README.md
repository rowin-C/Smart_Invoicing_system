<div >
<p>This is a implementation of Yolov5 model in solving the problem of billing of non-barcoded items </p>
<a href="https://docs.google.com/presentation/d/1MDFgdnaFxTpa-uoNY4t8NCto5PQ3rEBTnsMs_vVO-Mg/edit?usp=sharinghttps://docs.google.com/presentation/d/1MDFgdnaFxTpa-uoNY4t8NCto5PQ3rEBTnsMs_vVO-Mg/edit?usp=sharing">
Complete Information (Google slides)</a>

<p> Get this project Up and running</p>

<p> Backend Setup (in the root folder):</p>

```bash
pip install -r requirements.txt
```

<p> Frontend Setup (in Frontend UI
/smart_invoicing/):</p>

```bash
npm i
npm run build
```

<p> *npm run build ensures there are no breaking changes</p>

<p> Start the Backend (in the root folder):</p>

```bash
flask --app server run
```

<p> All the logic and code are in the server.py file (detection , API, excel management). One can improve it by making separate files, functions.</p>

<p> Start the Frontend  (in Frontend UI
/smart_invoicing/):</p>

```bash
npm run dev
```

<div class="container">
        <h1>Object Recognition System Documentation</h1>
        <h2>Overview</h2>
        <p>The Object Recognition System is designed to detect and manage various fruits and vegetables using a camera interface. The system is trained on a dataset consisting of 20 different fruits and vegetables. For detailed information on the training process, refer to the
        <a href="https://drive.google.com/drive/folders/1lAQsSeYI27QFk8lr7TGyp-AUIQZIDx3D?usp=sharing"><em>run/train</em></a> 
        folder.</p>
        
  <h2>Getting Started</h2>
  <ol>
            <li>
                <strong>Initialization</strong>
                <ul>
                    <li>Press the "Start" button to initiate the object recognition process.</li>
                </ul>
            </li>
            <li>
                <strong>Object Detection</strong>
                <ul>
                    <li>Place a vegetable or fruit, such as a tomato, in front of the camera.</li>
                    <li>The system will detect the item, and the interface will update from "No Detection" to "Detected".</li>
                </ul>
            </li>
            <li>
                <strong>Item Addition</strong>
                <ul>
                    <li>Click the "Detect" button to add the detected item to the item list on the front end.</li>
                </ul>
            </li>
        </ol>

  <h2>Workflow</h2>
  <ol>
            <li>
                <strong>Detect Item</strong>
                <ul>
                    <li>Bring an item in front of the camera.</li>
                    <li>Confirm the detection as indicated by the interface.</li>
                </ul>
            </li>
            <li>
                <strong>Add to List</strong>
                <ul>
                    <li>After detection, click "Detect" to add the item to the list.</li>
                    <li>The item will appear in the front-end list, where you can view and manage it.</li>
                </ul>
            </li>
            <li>
                <strong>Editing and Adding Items</strong>
                <ul>
                    <li>You can manually edit and add other items through the front-end interface as needed.</li>
                </ul>
            </li>
            <li>
                <strong>Generating Invoice</strong>
  <ul>
                    <li>Once all items are added and confirmed, click the "Pay" button.</li>
                    <li>An invoice will be generated, including a payment address and a QR code for convenience.</li>
  </ul>
  </li>
  <li>
                <strong>Printing Invoice</strong>
                <ul>
  <li>To print the invoice, click the "Print Invoice" button.</li>
  </ul>
  </li>
  </ol>

  <div class="additional-info">
            <h2>Additional Information</h2>
            <p>For a comprehensive understanding of the system's logic and workflow, refer to the 
            <a href="https://docs.google.com/presentation/d/1MDFgdnaFxTpa-uoNY4t8NCto5PQ3rEBTnsMs_vVO-Mg/edit?usp=sharinghttps://docs.google.com/presentation/d/1MDFgdnaFxTpa-uoNY4t8NCto5PQ3rEBTnsMs_vVO-Mg/edit?usp=sharing">
Google Slides</a>
             presentation provided.</p>
            <p>Ensure to check the <a href="https://drive.google.com/drive/folders/1lAQsSeYI27QFk8lr7TGyp-AUIQZIDx3D?usp=sharing"><em>run/train</em></a> folder for in-depth details on the training data and process.</p>
  </div>
</div>
