"""Echo-Sight: ask the Code of Student Conduct questions and hear grounded, cited answers.

    python app.py                        type questions, hear answers (Groq by default)
    python app.py --voice                push-to-talk: speak your questions
    python app.py --provider ollama      fully offline with a local model
    python app.py -q "Can I appeal?"     answer one question and exit
    python app.py --stair                add the STAIR table-of-contents scope gate

Say or type "repeat" to hear the last answer again, or press Enter on an empty line to quit.
"""

import argparse

from echosight.config import LLMConfig, RetrievalConfig
from echosight.llm import LLM
from echosight.rag import EchoSight
from echosight.retrieval import Retriever
from echosight.voice import Listener, SentenceStream, Speaker


def answer_aloud(assistant: EchoSight, speaker: Speaker, question: str, history, verbose: bool):
    stream = SentenceStream(speaker.say)

    def on_text(delta: str):
        print(delta, end="", flush=True)
        stream.feed(delta)

    print()
    answer = assistant.ask(question, on_text=on_text, history=history)
    stream.flush()
    print()
    if answer.cited:
        print("Sources: " + "; ".join(f"[{answer.hits.index(h) + 1}] {h.citation}" for h in answer.cited))
        speaker.say(answer.source_line)
    if verbose:
        t = answer.timings
        print(f"(retrieval {t['retrieval_s']:.2f}s, first token {t.get('ttft_s') or 0:.2f}s, "
              f"total {t['total_s']:.2f}s{', gated' if answer.gated else ''})")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--question", help="answer a single question and exit")
    parser.add_argument("--provider", help="groq (default), ollama or gemini")
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--voice", action="store_true", help="speak questions instead of typing them")
    parser.add_argument("--stair", action="store_true",
                        help="STAIR table-of-contents scope gate (routes with Groq gpt-oss-20b; needs GROQ_API_KEY)")
    parser.add_argument("--mute", action="store_true", help="print answers without speaking them")
    parser.add_argument("--rate", type=int, default=190, help="speech rate in words per minute")
    parser.add_argument("-v", "--verbose", action="store_true", help="show timings")
    args = parser.parse_args()

    speaker = Speaker(rate=args.rate, enabled=not args.mute)
    print("Loading Echo-Sight...")
    retrieval = RetrievalConfig(toc_router="groq:openai/gpt-oss-20b") if args.stair else RetrievalConfig()
    assistant = EchoSight(Retriever(retrieval), LLM(LLMConfig.from_env(args.provider, args.model)))
    if args.question:
        answer_aloud(assistant, speaker, args.question, None, args.verbose)
        speaker.wait()
        return

    listener = Listener() if args.voice else None
    speaker.say("Echo-Sight is ready. Ask a question about the Code of Student Conduct.")
    history, last = [], None
    while True:
        speaker.wait()
        question = listener.listen() if listener else input("\nQuestion (Enter to quit): ").strip()
        if listener:
            print(f"You asked: {question}")
        if not question:
            break
        if question.lower().strip(" .!?") in ("repeat", "say that again") and last:
            speaker.say(last.spoken)
            continue
        last = answer_aloud(assistant, speaker, question, history, args.verbose)
        if not last.refused:
            history.append((question, last.text))
    speaker.say("Goodbye.")
    speaker.wait()


if __name__ == "__main__":
    main()
