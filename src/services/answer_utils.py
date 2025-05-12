from services.llms.ai71_client import AI71Client
from services.llms.general_openai_client import GeneralOpenAIClient
from utils.logging_utils import get_logger

sys_prompt = """Rephrase this answer to be more concise while retaining all the information."""

stop_words = {'the', 'a', 'an', 'and', 'or',
              'but', 'is', 'are', 'was', 'were'}

log = get_logger('answer_utils')


def words_count(text: str) -> int:
    return len(text.split())


def condense_answer(llm_client: GeneralOpenAIClient, answer: str, words_limit=300) -> str:
    """
    Condense the answer by removing unnecessary whitespace and newlines.
    """
    words = answer.split()
    original_words_count = len(words)
    if original_words_count <= words_limit:
        return answer

    answer, _ = llm_client.complete_chat_once(
        message=sys_prompt, system_message=answer)

    # condense the answer
    words = answer.split()
    if len(words) > words_limit:
        # drop all stop words from the answer
        words = [word for word in words if word.lower() not in stop_words]
        answer = ' '.join(words)

    log.info(f"Condensed answer", original_words_count=original_words_count,
             words_count=len(words), words_limit=words_limit)
    return answer


if __name__ == "__main__":
    # Test the condense_answer function
    llm_client = AI71Client()
    answer = """One-coat stucco systems, also known as thin-coat stucco systems, have several potential drawbacks and limitations when used on exterior walls. These include:

1. Less impact resistance: One-coat stucco systems are less impact resistant than traditional three-coat stucco. This means they may be more susceptible to damage from impacts, such as from falling objects or high winds.

2. Thinner profile: With a thickness of only 3/8-inch, one-coat systems are less able to hide irregularities in the framing and are more likely to have thin spots that are prone to problems.

3. Not completely waterproof: Although one-coat stucco systems are more waterproof and less prone to shrinkage cracking than traditional stucco, they are not completely waterproof. Over time, water can find its way in at joints, penetrations, or cracks, and the synthetic stucco will be slower to dry out than the more permeable traditional stucco.

4. Proprietary systems: Each one-coat stucco system is proprietary and must be installed according to the manufacturer's approved specs and details. This can limit the choice of products and installers, and if not installed correctly, warranties may be voided and code approvals invalidated.

5. Potential moisture issues: Similar to other exterior wall systems, one-coat stucco systems can trap moisture behind the highly water-resistant material, which can lead to rot, mold, and pest infestations if not properly maintained.

6. Aesthetics: While one-coat stucco systems can provide a smoother finish and more uniform color, some homeowners may prefer the more muted and variable color of traditional cement stucco.

7. Installation complexity: Although one-coat stucco systems are designed to simplify the application process, they still require skilled installation to ensure proper adhesion and long-term performance.

In summary, while one-coat stucco systems offer some advantages, such as faster application and a smoother finish, they also have limitations and potential drawbacks that should be considered when choosing an exterior wall system for a home."""
    condensed_answer = condense_answer(llm_client, answer)
    print(condensed_answer)
