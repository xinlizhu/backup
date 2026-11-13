# Appendix: Prompts for QDiag Data Construction

We generate text edits using three separate text-only requests per source instruction: ADD, DELETE, and SUBSTITUTE. Each request considers object, attribute, and predicate edits, yielding at most nine candidates. We then review the candidates jointly at the group level using the reference image(s) and supplied positive retrieval images. Both stages use `gpt-5.6-terra`.

Below we present the task instructions, omitting image-count and serialization details for readability. The review criteria apply to the supplied positive images. ADD and SUBSTITUTE form the strict-negative subset; DELETE yields weaker queries and is reported separately.

## A. Text-only generation

Each request supplies the original instruction and the designated edit operation. The model returns one proposal or `null` for each semantic target, together with the edited text and its before/after spans.

### ADD

```text
Generate text-only ADD hard-negative instructions. You receive exactly one positive instruction and
no images. Return the object, attribute, and predicate slots; each slot is one proposal or null.

For each proposal, insert exactly one contiguous after_span into the original instruction and change
nothing else. Set before_span=null. Removing that exact after_span from negative_text must recover
the original source_instruction apart from whitespace and terminal punctuation.

- object: add one named entity or entity-presence constraint.
- attribute: add one property to an entity already named or unambiguously referenced.
- predicate: add one action or relation over entities already named or referenced.

The result must be fluent, natural, close to the original, and plausibly difficult. Do not use
images, claim visual evidence, rewrite the sentence, restate existing meaning, or add multiple
constraints. Use null when a safe single insertion is unavailable. Return only the structured
response.
```

### DELETE

```text
Generate text-only DELETE hard-negative instructions. You receive exactly one positive instruction
and no images. Return the object, attribute, and predicate slots; each slot is one proposal or null.

For each proposal, delete exactly one contiguous before_span from the original instruction and
change nothing else. Set after_span=null. Deleting that exact span must recover negative_text apart
from whitespace and terminal punctuation. The result must remain fluent, meaningful, and strictly
weaker.

Count semantic constraints rather than raw tokens. A before_span may include grammatical glue that
belongs only to the deleted constraint, such as an article, preposition, comma, or coordinator
"and". For example, "brown and white animals" may become "white animals" by deleting "brown and";
that is one attribute deletion.

- object: remove one entity constraint while leaving a meaningful query.
- attribute: remove one property while retaining its entity.
- predicate: remove one action/relation while retaining its arguments.

Do not delete bundled independent constraints and do not repair or rewrite the remaining sentence.
Use null when no safe deletion exists. Return only the structured response.
```

### SUBSTITUTE

```text
Generate text-only SUBSTITUTE hard-negative instructions. You receive exactly one positive
instruction and no images. Return the object, attribute, and predicate slots; each slot is one
proposal or null.

For each proposal, replace exactly one contiguous before_span from the original instruction with
exactly one contiguous after_span and change nothing else. Both spans must be explicit and different.
Applying that one replacement must recover negative_text apart from whitespace and terminal
punctuation.

- object: replace one entity identity or category.
- attribute: replace one property value while retaining its entity.
- predicate: replace one action or relation while retaining its arguments and semantic scope.

The result must be fluent, natural, close to the original, and plausibly difficult. Do not use
images, invent visual evidence, replace multiple values, switch semantic axes, or rewrite the
sentence. Use null when a safe single replacement is unavailable. Return only the structured
response.
```

## B. Vision-grounded review

The review request supplies the group's original instructions and all generated candidates, including each candidate's source text, edit operation, semantic target, and proposed spans. It also supplies the reference image(s) and positive retrieval images. The model returns an accept/reject decision and a concise reason for each candidate.

```text
Validate every supplied hard-negative text for one PinPoint paraphrase group in exactly one pass.
The supplied images are labeled as REFERENCES or POSITIVE retrieval results.
Official negative images are not included in this request. Give exactly one verdict for every negative_id; do not omit, duplicate,
invent, or rewrite candidates.

First check every candidate against its own source_text:
- it is fluent, natural, close to source_text, and preserves all unedited meaning;
- it performs exactly one stated edit and matches semantic_target;
- object means entity identity/presence, attribute means a property of an existing entity, and
  predicate means an action/relation with the same arguments and scope.
- before_span and after_span are untrusted generator claims. Verify them against source_text and
  negative_text yourself. Reject as not_single_edit when the text is unchanged, the stated span
  operation cannot reconstruct negative_text, or more than one semantic constraint changed.

ADD and SUBSTITUTE:
- use the REFERENCE image(s) to resolve phrases such as "this", "same", <img1>, and <img2>;
- accept only when the new fact is clearly false for EVERY SUPPLIED POSITIVE image;
- if any supplied POSITIVE supports or remains compatible with the new fact, reject as
  positive_compatible;
- if any required fact cannot be judged reliably from the references and all supplied positives,
  reject as not_visually_verifiable;
- difficulty must come from a subtle single edit, not from ambiguity or hidden evidence.

DELETE:
- do not require the weaker text to contradict positive images;
- accept when it removes exactly one meaningful constraint, preserves everything else, and produces
  a fluent, meaningful, strictly weaker query;
- count semantic constraints, not raw tokens. Removing grammatical glue belonging only to the
  deleted constraint is still one edit. For example, "brown and white animals" -> "white animals"
  is one attribute DELETE;
- reject bundled deletions, a wrong semantic axis, changed surviving meaning, or broken grammar.

For rejection choose exactly one failure_code from the response schema and state the decisive
reason concisely. For acceptance use failure_code=null. Return only the structured response.
```
