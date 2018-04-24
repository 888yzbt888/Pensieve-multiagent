function postHB(){
    console.log('begin');
    var http = new XMLHttpRequest();
    //the message
    var data={
        'heartbeat':1
    }
    data=JSON.stringify(data);
    http.open("POST", "http://192.168.1.198:8333", true);//my IP address
    http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    http.onreadystatechange = function() {
        console.log('onreadystatechange');
        if (http.readyState == 4 && http.status == 200) {
            console.log(http.responseText);
        }
        else {
            console.log('readyState=' + http.readyState + ', status: ' + http.status);
        }
    }
    console.log('sending...')
    http.send(data);
    console.log('end');
}
setInterval(postHB,5000);
