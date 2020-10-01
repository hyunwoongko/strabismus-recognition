let ner_model = false;

function ner() {
    const urls = $("#pororo_ner_urls")
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const load_model = urls.data("load_model");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + load_model

    $("#pororo_ner_button").hide();
    $(".loading").show();
    $("#pororo_ner_result_body").empty()
    $(".loading-text").text("모델을 불러오고 있습니다...");
    $(".loading-text").show();
    // 버튼 못누르게 숨기기 + 로딩 띄우기

    if (ner_model === false) {
        $.ajax({
            //
            url: request_url,
            type: "GET",
            dataType: "json",
            success: function (data) {
                console.log("Model Load Status : " + data["status"])
                ner_model = true;
                ner_predict(urls)
            },

            error: function (request, status, error) {
                console.log(error);
                $("#pororo_ner_button").show();
                $(".loading").hide();
                $(".loading-text").hide();
                alert("모델을 로드하지 못했어요.");
            }
        });
    } else {
        ner_predict(urls);
    }
}

function ner_predict(urls) {
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const pororo_ner_predict = urls.data("pororo_ner_predict");

    let text = $("#pororo_ner_prompt").val()
        .replaceAll("\n", " ")
        .replaceAll("?", ".");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + pororo_ner_predict + "/" + text

    $(".loading-text")
        .text("개체명을 인식하고 있습니다...");

    $.ajax({
        //
        url: request_url,
        type: "GET",
        dataType: "json",
        success: function (data) {
            $("#pororo_ner_result_body")
                .append(
                    "<li>문장 : " + text + "</li>" +
                    "<li>인식 결과 : " + data["prediction"] + "</li>"
                )
            $("#pororo_ner_button").show();
            $(".loading").hide();
            $(".loading-text").hide();

        },
        error: function (request, status, error) {
            console.log(error);
            $("#pororo_ner_button").show();
            $(".loading").hide();
            $(".loading-text").hide();
            alert("개체명 인식에 실패했습니다.")
        }
    });
}
