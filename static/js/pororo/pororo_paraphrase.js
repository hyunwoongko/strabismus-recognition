let paraphrase_model = false;

function paraphrase() {
    const urls = $("#pororo_paraphrase_urls")
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const load_model = urls.data("load_model");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + load_model

    $("#pororo_paraphrase_button").hide();
    $(".loading").show();
    $("#pororo_paraphrase_result_body").empty()
    $(".loading-text").text("모델을 불러오고 있습니다...");
    $(".loading-text").show();
    // 버튼 못누르게 숨기기 + 로딩 띄우기

    if (paraphrase_model === false) {
        $.ajax({
            //
            url: request_url,
            type: "GET",
            dataType: "json",
            success: function (data) {
                console.log("Model Load Status : " + data["status"])
                paraphrase_model = true;
                paraphrase_predict(urls)
            },

            error: function (request, status, error) {
                console.log(error);
                $("#pororo_paraphrase_button").show();
                $(".loading").hide();
                $(".loading-text").hide();
                alert("모델을 로드하지 못했어요.");
            }
        });
    } else {
        paraphrase_predict(urls);
    }
}

function paraphrase_predict(urls) {
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const pororo_paraphrase_predict = urls.data("pororo_paraphrase_predict");

    let text = $("#pororo_paraphrase_prompt").val()
        .replaceAll("\n", " ")
        .replaceAll("?", ".");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + pororo_paraphrase_predict + "/" + text

    $(".loading-text")
        .text("패러프레이즈를 생성하고 있습니다...");

    $.ajax({
        //
        url: request_url,
        type: "GET",
        dataType: "json",
        success: function (data) {
            $("#pororo_paraphrase_result_body")
                .append(
                    "<li>문장 : " + text + "</li>" +
                    "<li>생성 결과 : " + data["prediction"] + "</li>"
                )
            $("#pororo_paraphrase_button").show();
            $(".loading").hide();
            $(".loading-text").hide();

        },
        error: function (request, status, error) {
            console.log(error);
            $("#pororo_paraphrase_button").show();
            $(".loading").hide();
            $(".loading-text").hide();
            alert("패러프레이즈 생성에 실패했습니다.")
        }
    });
}
