let sentiment_analysis_model = false;

function sentiment_analysis() {
    const urls = $("#pororo_sentiment_analysis_urls")
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const load_model = urls.data("load_model");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + load_model

    $("#pororo_sentiment_analysis_button").hide();
    $(".loading").show();
    $("#pororo_sentiment_analysis_result_body").empty()
    $(".loading-text").text("모델을 불러오고 있습니다...");
    $(".loading-text").show();
    // 버튼 못누르게 숨기기 + 로딩 띄우기

    if (sentiment_analysis_model === false) {
        $.ajax({
            //
            url: request_url,
            type: "GET",
            dataType: "json",
            success: function (data) {
                console.log("Model Load Status : " + data["status"])
                sentiment_analysis_model = true;
                sentiment_analysis_predict(urls)
            },

            error: function (request, status, error) {
                console.log(error);
                $("#pororo_sentiment_analysis_button").show();
                $(".loading").hide();
                $(".loading-text").hide();
                alert("모델을 로드하지 못했어요.");
            }
        });
    } else {
        sentiment_analysis_predict(urls);
    }
}

function sentiment_analysis_predict(urls) {
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const pororo_sentiment_analysis_predict = urls.data("pororo_sentiment_analysis_predict");

    let text = $("#pororo_sentiment_analysis_prompt").val()
        .replaceAll("\n", " ")
        .replaceAll("?", ".");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + pororo_sentiment_analysis_predict + "/" + text

    $(".loading-text")
        .text("감정을 분류하고 있습니다...");

    $.ajax({
        //
        url: request_url,
        type: "GET",
        dataType: "json",
        success: function (data) {
            $("#pororo_sentiment_analysis_result_body")
                .append(
                    "<li>문장 : " + text + "</li>" +
                    "<li>분류 결과 : " + data["prediction"] + "</li>"
                )
            $("#pororo_sentiment_analysis_button").show();
            $(".loading").hide();
            $(".loading-text").hide();

        },
        error: function (request, status, error) {
            console.log(error);
            $("#pororo_sentiment_analysis_button").show();
            $(".loading").hide();
            $(".loading-text").hide();
            alert("감정 분류에 실패했습니다.")
        }
    });
}
