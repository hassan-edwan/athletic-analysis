// Root-cause diagnostics for sprint form faults (port of diagnostics.py).
// The knowledge base is transcribed verbatim from the Python core — that file
// is the reference; edit both together.

import Foundation

public struct Diagnosis: Sendable {
    public var key: String
    public var deviation: String  // "low" | "high"
    public var title: String
    public var technicalCauses: [String]
    public var muscleFactors: [String]
    public var drills: [String]
    public var phaseNote: String
    public var source: String

    init(key: String, deviation: String, title: String,
         technicalCauses: [String], muscleFactors: [String], drills: [String],
         phaseNote: String = "", source: String = "") {
        self.key = key
        self.deviation = deviation
        self.title = title
        self.technicalCauses = technicalCauses
        self.muscleFactors = muscleFactors
        self.drills = drills
        self.phaseNote = phaseNote
        self.source = source
    }
}

public enum Diagnostics {
    static let mannSrc = "front-side mechanics / touchdown position (Mann sprint model)"
    static let gctSrc = "GCT: elite ~0.09 s at Vmax (Sides 2018; Nagahara)"
    static let trunkSrc = "trunk ~45° block exit → vertical at Vmax (auptimo; World Athletics)"
    static let cadSrc = "cadence rises with level; ~4–5 Hz elite (Nagahara)"
    static let kneeSrc = "stiffer, more extended touchdown leg with level (Mann)"

    struct KBKey: Hashable {
        let key: String
        let deviation: String
        let phase: String  // "" = generic
    }

    static let kb: [KBKey: Diagnosis] = {
        var out: [KBKey: Diagnosis] = [:]
        func add(_ phase: String, _ d: Diagnosis) {
            out[KBKey(key: d.key, deviation: d.deviation, phase: phase)] = d
        }

        add("", Diagnosis(
            key: "overstride", deviation: "high", title: "Overstriding",
            technicalCauses: [
                "Reaching for the ground — the foot extends forward late in swing "
                + "instead of stepping down under a rising knee",
                "Compensating for a low step rate by seeking stride length out front",
                "Backside-dominant leg cycle: the recovery leg arrives late, so the "
                + "foot lands ahead of the hips and brakes every step",
            ],
            muscleFactors: [
                "Weak hamstrings — unable to decelerate the swinging shank before "
                + "touchdown, letting the foot fly out front",
                "Weak glutes — limited hip-extension power, so length is sought in "
                + "front of the body instead of behind it",
            ],
            drills: [
                "A-skips and dribble runs with a 'step down and back' cue",
                "Wicket / mini-hurdle runs at fixed spacing to raise cadence",
                "Nordic hamstring curls (eccentric leg deceleration)",
                "Hip thrusts / glute bridges (terminal hip extension)",
            ],
            source: mannSrc))

        add("", Diagnosis(
            key: "contact_ms", deviation: "high", title: "Long ground contact",
            technicalCauses: [
                "The leg yields at the ankle/knee on contact instead of behaving "
                + "like a stiff spring",
                "Touchdown too far ahead of the hips lengthens the braking half of "
                + "stance",
                "Pushing too long behind the body instead of cycling the leg through",
            ],
            muscleFactors: [
                "Low reactive/tendon stiffness in the calf–Achilles complex",
                "Weak eccentric quad control, allowing knee collapse in early stance",
                "General maximal-strength deficit — force takes longer to produce",
            ],
            drills: [
                "Pogo hops and ankling (stiff ankle, minimal ground time)",
                "Straight-leg bounds; low depth drops with an instant rebound",
                "Heavy calf raises and calf isometrics",
                "Sled pushes / hill sprints for drive-phase force production",
            ],
            source: gctSrc))

        add("", Diagnosis(
            key: "contact_ms", deviation: "low", title: "Unusually short ground contact",
            technicalCauses: [
                "Most often a measurement artifact — verify the capture FPS setting "
                + "matches how the clip was filmed (slow-motion footage is a common "
                + "mismatch)",
                "If the timing is real: contacts cut short before the push finishes, "
                + "bleeding propulsion",
            ],
            muscleFactors: [
                "Rarely a strength issue — rule out data problems first",
            ],
            drills: [
                "Re-check Capture FPS…; re-film at 60–240 fps if possible",
                "If real: cue 'finish each push' — complete hip extension at toe-off",
            ],
            source: gctSrc))

        add("", Diagnosis(
            key: "thigh", deviation: "low",
            title: "Low knee lift (poor front-side mechanics)",
            technicalCauses: [
                "Backside-dominant cycle — the heel kicks up behind instead of the "
                + "knee stepping over the opposite thigh",
                "Anterior pelvic tilt tips the pelvis forward and caps how high the "
                + "thigh can rise",
                "Excess forward trunk lean at top speed blocks the thigh's path",
            ],
            muscleFactors: [
                "Weak hip flexors (iliopsoas) at high hip-flexion velocity",
                "Weak anterior core — the pelvis is not held stable as the knee "
                + "drives up",
                "Tight hip extensors/hamstrings limiting front-side range",
            ],
            drills: [
                "Wall drill / A-march holds with a tall pelvis",
                "A-skips pausing at the top knee position",
                "Banded or cable hip-flexor marches; hanging knee raises",
                "Dead bugs and hollow holds (pelvic control)",
            ],
            phaseNote: "Assessed at max velocity, where front-side mechanics matter most.",
            source: mannSrc))

        add("", Diagnosis(
            key: "cadence", deviation: "low", title: "Low step rate",
            technicalCauses: [
                "Long ground contacts leave the legs waiting on the ground",
                "Overstriding — each stride is reaching, not turning over",
                "Passive recovery: the swing leg is not actively pulled through",
            ],
            muscleFactors: [
                "Low reactive/elastic strength — slow ground rebound caps turnover",
                "Weak hip flexors slowing the recovery leg's punch-through",
            ],
            drills: [
                "Fast-leg drills and dribble runs (quick, punchy contacts)",
                "Wicket runs at slightly-tighter-than-natural spacing",
                "Pogo hops and low hops for rebound quickness",
            ],
            source: cadSrc))

        add("", Diagnosis(
            key: "cadence", deviation: "high", title: "Very high step rate",
            technicalCauses: [
                "Strides cut short — the push ends before full hip extension, so "
                + "steps are quick but empty",
                "Spinning: cadence substitutes for force because each contact "
                + "produces little propulsion",
            ],
            muscleFactors: [
                "Strength/power deficit — not enough force per contact, compensated "
                + "with turnover",
            ],
            drills: [
                "Bounds and single-leg hops for force per contact",
                "Cue 'push the ground away and finish each step'",
                "Heavy strength work: squats, hip thrusts, step-ups",
            ],
            source: cadSrc))

        add("", Diagnosis(
            key: "knee_strike", deviation: "low", title: "Knee collapsing at touchdown",
            technicalCauses: [
                "The leg lands bent and keeps bending — it absorbs instead of "
                + "rebounding",
                "Touchdown too far ahead of the hips forces the knee to buckle "
                + "under braking load",
            ],
            muscleFactors: [
                "Weak eccentric quad strength at touchdown",
                "Low leg/ankle stiffness — the limb cannot hold a spring-like shape",
            ],
            drills: [
                "Depth landings holding a tall, stiff leg",
                "Stiffness bounds and pogo hops",
                "Split squats and step-downs (eccentric quad control)",
            ],
            source: kneeSrc))

        add("", Diagnosis(
            key: "knee_strike", deviation: "high", title: "Landing on a locked leg",
            technicalCauses: [
                "The leg is fully extended and rigid at touchdown — no give to "
                + "absorb and return energy, and impact travels up the joint chain",
                "Often paired with overstriding: a straight leg reaching out front",
            ],
            muscleFactors: [
                "Usually technical rather than a weakness — but stiff hips/ankles "
                + "can force the knee to stay locked",
            ],
            drills: [
                "Cue a soft, 'spring-loaded' knee at contact, landing closer to the "
                + "hips",
                "A-skips focusing on an active down-and-back touchdown",
                "Ankle mobility work if the ankle cannot dorsiflex under load",
            ],
            source: kneeSrc))

        // --- trunk: meaning depends on phase ---

        add("drive", Diagnosis(
            key: "trunk", deviation: "low", title: "Standing up out of the start",
            technicalCauses: [
                "Popping upright in the first steps instead of holding the forward "
                + "projection of the drive phase",
                "First steps too short/choppy, forcing the body to rise early",
            ],
            muscleFactors: [
                "Weak glutes and quads — not strong enough to hold the low drive "
                + "position while pushing",
                "Weak core — the torso cannot hold a rigid line at 45°",
            ],
            drills: [
                "Sled pushes and wall drills holding the drive angle",
                "Hill sprints (the slope enforces forward lean)",
                "Front planks and dead bugs for a rigid torso line",
            ],
            phaseNote: "In the drive phase a strong forward lean is the goal, not a fault.",
            source: trunkSrc))

        add("drive", Diagnosis(
            key: "trunk", deviation: "high", title: "Over-leaning out of the start",
            technicalCauses: [
                "Leaning past what the legs can support — the athlete is falling, "
                + "not driving, and the feet chase the torso",
                "Head/eyes down, folding at the waist rather than leaning as one line",
            ],
            muscleFactors: [
                "Weak core/hip extensors — cannot hold a straight body line, so the "
                + "waist folds to fake the lean",
            ],
            drills: [
                "Falling starts: lean as one rigid line and let the fall start the run",
                "Sled pushes with the hips in line with the shoulders",
                "Back extensions and hip hinges for a solid posterior line",
            ],
            source: trunkSrc))

        add("max velocity", Diagnosis(
            key: "trunk", deviation: "low", title: "Leaning back at top speed",
            technicalCauses: [
                "Running behind the hips — the pelvis drifts forward while the torso "
                + "stays back, often late in a rep as fatigue sets in",
                "Over-cueing 'run tall' into a backward lean",
            ],
            muscleFactors: [
                "Weak anterior core letting the pelvis tilt and the ribcage flare",
                "Tight hip flexors pulling the lumbar spine into extension",
            ],
            drills: [
                "Cue 'eyes level, hips under shoulders'",
                "Hollow holds and dead bugs (rib-to-pelvis connection)",
                "Hip-flexor mobility work (couch stretch)",
            ],
            source: trunkSrc))

        add("max velocity", Diagnosis(
            key: "trunk", deviation: "high", title: "Excess forward lean at top speed",
            technicalCauses: [
                "Still 'driving' when the body should be upright — the forward lean "
                + "kills front-side mechanics and forces the legs to catch the body",
                "Folding at the waist under fatigue",
            ],
            muscleFactors: [
                "Weak spinal erectors/glutes — the torso droops as the rep goes on",
                "Weak core failing to hold the pelvis level under speed",
            ],
            drills: [
                "Tall-running drills: A-runs with 'hips tall, chest proud'",
                "Back extensions, reverse hypers, RDLs for the posterior chain",
                "Timed runs cut before posture breaks down, building volume gradually",
            ],
            source: trunkSrc))

        add("acceleration", Diagnosis(
            key: "trunk", deviation: "low", title: "Rising too early in the transition",
            technicalCauses: [
                "Snapping upright instead of letting the trunk rise gradually as "
                + "speed builds",
            ],
            muscleFactors: [
                "Weak glutes/quads cutting the drive short — standing up is easier "
                + "than continuing to push",
            ],
            drills: [
                "Sled marches into free running with a gradual rise",
                "Hill starts transitioning onto the flat",
            ],
            source: trunkSrc))

        add("acceleration", Diagnosis(
            key: "trunk", deviation: "high", title: "Staying crouched too long",
            technicalCauses: [
                "Holding the start-lean after the speed that justified it — the low "
                + "torso now blocks knee lift and shortens each stride",
            ],
            muscleFactors: [
                "Usually technical/habitual rather than a strength deficit",
            ],
            drills: [
                "Cue a smooth rise: 'let the track come up to you' over 10–20 m",
                "A-skip transitions from a lean into tall running",
            ],
            source: trunkSrc))

        return out
    }()

    /// Root-cause entry for a fault finding; nil for in-range findings,
    /// unknown keys, or artifact-direction cues.
    public static func diagnose(_ finding: FormFinding) -> Diagnosis? {
        guard finding.severity != .good, !finding.key.isEmpty,
              !finding.deviation.isEmpty else { return nil }
        return kb[KBKey(key: finding.key, deviation: finding.deviation,
                        phase: finding.phase)]
            ?? kb[KBKey(key: finding.key, deviation: finding.deviation, phase: "")]
    }

    public static func allDiagnoses() -> [Diagnosis] {
        Array(kb.values)
    }
}
