let short_answer_model = false;

function short_answer() {
    const urls = $("#brinqa_short_answer_urls")
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const load_model = urls.data("load_model");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + load_model

    $("#brainqa_short_answer_button").hide();
    $(".loading").show();
    $("#brainqa_short_answer_result_body").empty()
    $(".loading-text").text("모델을 불러오고 있습니다...");
    $(".loading-text").show();
    // 버튼 못누르게 숨기기 + 로딩 띄우기

    if (short_answer_model === false) {
        $.ajax({
            //
            url: request_url,
            type: "GET",
            dataType: "json",
            success: function (data) {
                console.log("Model Load Status : " + data["status"])
                short_answer_model = true;
                qg_answer(urls)
            },

            error: function (request, status, error) {
                console.log(error);
                $("#brainqa_short_answer_button").show();
                $(".loading").hide();
                $(".loading-text").hide();
                alert("모델을 로드하지 못했어요.");
            }
        });
    } else {
        qg_answer(urls);
    }
}

function qg_answer(urls) {
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const brainqa_qg_answer = urls.data("brainqa_qg_answer");

    let context = $("#brainqa_short_answer_prompt").val()
        .replaceAll("\n", " ")
        .replaceAll("?", ".");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + brainqa_qg_answer + "/" + context

    $.ajax({
        //
        url: request_url,
        type: "GET",
        dataType: "json",
        success: function (data) {
            qg_question(
                data["summary"],
                data["answers"],
                urls
            )
        },

        error: function (request, status, error) {
            console.log(error);
            $("#brainqa_short_answer_button").show();
            $(".loading").hide();
            $(".loading-text").hide();
            alert("문장 생성에 실패했습니다.")
        }
    });
}


function qg_question(summary, answers, urls) {
    const node_id = urls.data("node_id");
    const port = urls.data("port");
    const brainqa_qg_question = urls.data("brainqa_qg_question");

    const request_url = "http://" + node_id + ".dakao.io:" + port + "/"
        + brainqa_qg_question + "/" + summary + "/"

    let response_count = 0;
    let error_count = 0;

    $(".loading-text")
        .text("문제를 만들고 있습니다... (" + response_count + "/" + answers.length + ")");

    for (const answer of answers) {
        console.log(request_url + answer);

        $.ajax({
            //
            url: request_url + answer,
            type: "GET",
            dataType: "json",
            success: function (data) {
                response_count++;
                $(".loading-text")
                    .text("문제를 만들고 있습니다... (" + response_count + "/" + answers.length + ")");

                $("#brainqa_short_answer_result_body")
                    .append(
                        "<li>문제 " + response_count +
                        "<ul>" +
                        "<li>Q : " + data["question"] + "</li>" +
                        "<li>A : " + answer + "</li>" +
                        "</ul>" +
                        "</li>"
                    )

                if (response_count === answers.length) {
                    $("#brainqa_short_answer_button").show();
                    $(".loading").hide();
                    $(".loading-text").hide();
                }
            },
            error: function (request, status, error) {
                console.log(error);
                $("#brainqa_short_answer_button").show();
                $(".loading").hide();
                $(".loading-text").hide();

                if (error_count === 0) {
                    alert("문장 생성에 실패했습니다.")
                    error_count++;
                }
            }
        });
    }
}
