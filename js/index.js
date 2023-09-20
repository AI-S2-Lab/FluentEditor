$(function(){
    
	
	$("#submit_register").on("click",function(e){
		e.preventDefault();
		
		if(flag == "true"){
			var formData = $("#register_form").serialize();
			formData = decodeURIComponent(formData);   // 解码，防止中文乱码
			console.log(formData);
			$.ajax({
				type: "POST", // 数据提交类型
				url: "http://39.104.64.144:81/ttsChanllengeRegister", // 发送地址
				data: formData, //发送数据
				success : function(result){
					console.log(result);
					if(result.code == 0){
						alert("注册成功！");
					}else if(result.code == 1){
						alert(result.note);
					}else{
						alert("注册失败，请再次尝试。（若多次失败，请联系竞赛主办方）")
					}
				},
				error : function(e){
					console.log(e.status);
					console.log(e.responseText);
					alert("申请上传失败，请联系网站管理员处理");
				}
			});
		}else{
			alert("请填写完整信息");
		}
	})
		

});

