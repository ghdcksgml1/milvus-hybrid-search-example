import {OpenAI} from "openai";
import {z} from "zod";
import {zodResponsesFunction} from "openai/helpers/zod";
import {ResponseStream} from "openai/lib/responses/ResponseStream";
import {ResponseInput} from "openai/resources/responses/responses";

const openai = new OpenAI();

///////////////////
// function call //
///////////////////
const RAG_NAME = 'hiworks_manual_retrieval';
const Rag = z.object({
  user_question: z
    .string()
    .describe("사용자의 구체적인 질문 또는 문의사항 (예: '휴가신청은 어떻게 하나요?', '근태 현황을 확인하고 싶어요')")
});
const RagResponse = z
  .array(z.string())
  .describe("하이웍스 매뉴얼에서 검색된 관련 정보의 목록");
const ragClient = async (params: z.infer<typeof Rag>): Promise<string[]> => {
  const url = encodeURI(`${process.env.MANUAL_URL}?query=${params.user_question}&limit=5`);
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    }
  });
  const parsedResponse = RagResponse.safeParse(await response.json())
  if (!parsedResponse.success) {
    throw new Error(`RAG API error: ${parsedResponse.error.message}`);
  }
  return parsedResponse.data;
}

///////////////////
// event handler //
///////////////////
const openaiEventHandler = async (result: ResponseStream) => {
  result
    .on('response.output_text.delta', (event) => {
      process.stdout.write(event.delta);
    })
    .on('response.reasoning_text.done', (event) => {
      process.stdout.write(`
<summary>
${event.text}
</summary>
`);
    });

  await result.done();
  return result.finalResponse();
}

///////////////////
// openai runner //
///////////////////
const runner = async (userInput: string) => {
  let input: ResponseInput = [{
    role: 'user',
    content: userInput,
  }]
  let currentRequest = 0;
  let previousResponseId: string | undefined = undefined;
  const limitRequest = 5;
  const references: string[] = [];

  while (currentRequest++ < limitRequest) {
    const functionCall = zodResponsesFunction({
      name: RAG_NAME,
      parameters: Rag,
      function: (params: z.infer<typeof Rag>) => ragClient(params),
      description: '하이웍스 매뉴얼에서 사용자의 질문과 관련된 정보를 검색합니다.'
    });
    const result = openai.responses.stream({
      previous_response_id: previousResponseId,
      instructions: `사용자의 질문에 대해 하이웍스 매뉴얼을 기반으로 객관적인 정보를 제공합니다.\n질문에 답할 수 없는 경우, 모른다고 대답하세요.`,
      input: input,
      model: 'gpt-5-mini',
      tools: [functionCall],
      reasoning: {
        effort: 'minimal',
      },
      store: true,
      stream: true,
    });

    let requireRecursiveCall = false;
    const finalResponse = await openaiEventHandler(result);

    previousResponseId = finalResponse.id;
    input = [];

    if (finalResponse.output) {
      await Promise.all(
        finalResponse.output.map(async (item) => {
          if (item.type === 'function_call') {
            if (item.parsed_arguments === null) {
              item.parsed_arguments = {};
            }
            const args = JSON.parse(item.arguments) as z.infer<typeof Rag>;
            const ragResponse = await ragClient(args);
            references.push(...ragResponse);
            input.push({
              type: 'function_call_output',
              call_id: item.call_id,
              output: JSON.stringify(ragResponse),
            });
            requireRecursiveCall = true;
          }
        })
      );
    }

    if (!requireRecursiveCall) break;
  }
  console.log(`\n\n===참고한 자료===\n${references.map((reference) => `[${reference.slice(0, 20)}...]`).join(',\n')}`);
};

///////////////////
// openai caller //
///////////////////
runner('휴가는 어떻게 신청해?').catch(console.error);
