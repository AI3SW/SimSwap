const content_div_id_list = [
    "post_image_content_div"
];

var toggle_content = function (content_div_id) {
    for (let index = 0; index < content_div_id_list.length; index++) {
        if (content_div_id === content_div_id_list[index]) {

            $("#" + content_div_id_list[index]).removeClass("d-none");
        } else {
            $("#" + content_div_id_list[index]).addClass("d-none");
        }
    }
};

var get_base64_string_from_img = function (selector) {
    let data_url = $(selector).prop("src");
    return data_url.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");
};

var loadSourceImage = function (event) {
    var reader = new FileReader();
    reader.onload = function () {
        $("#src_img").prop("src", reader.result);
        $("#src_img").removeClass("d-none");
    };
    reader.readAsDataURL(event.target.files[0]);
};

var loadRefImage = function (event) {
    var reader = new FileReader();
    reader.onload = function () {
        $("#ref_img").prop("src", reader.result);
        $("#ref_img").removeClass("d-none");
    };
    reader.readAsDataURL(event.target.files[0]);
};

window.onload = function () {
    $("#transfer_images_button").click(function () {
        let data = {
            src_img: get_base64_string_from_img("#src_img"),
            ref_img: get_base64_string_from_img("#ref_img")
        };

        $.ajax("predict", {
            data: JSON.stringify(data),
            contentType: 'application/json',
            type: 'POST',
            success: function (result) {
                toggle_content("post_image_content_div");
                $("#post_image_content_img").prop("src", "data:image/png;base64, " + result["output_img"]);
            }
        });
    });
};
