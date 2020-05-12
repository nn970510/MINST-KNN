var loadFile = function(event) {
	var imageDiv = document.getElementById('displayImgDiv');
	for(var i = 0; i < event.target.files.length; i++) {
	    var image =new Image();
	    image.src = URL.createObjectURL(event.target.files[i]);
	    imageDiv.appendChild(image);
    }
};