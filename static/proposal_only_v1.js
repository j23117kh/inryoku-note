(() => {
    "use strict";

    function normalize(value) {
        return String(value || "").replace(/\s+/g, "");
    }

    function keepProposalOnly() {
        const buttons = Array.from(
            document.querySelectorAll("button")
        );

        const proposal = buttons.find(button => {
            const text = normalize(button.textContent);
            return text.includes("統合提案") &&
                   text.includes("時間");
        });

        // Do not change anything if the target is missing.
        if (!proposal) {
            console.warn("Proposal button not found");
            return;
        }

        const removeLabels = [
            "提案：意味＋思考関係",
            "旧方式：自動系列推定",
            "承認済み関係だけ",
            "従来の統合方式",
            "意味検索のみ"
        ];

        let count = 0;

        for (const button of buttons) {
            if (button === proposal) continue;

            const text = normalize(button.textContent);

            if (removeLabels.some(label =>
                text.includes(normalize(label))
            )) {
                button.hidden = true;
                button.style.setProperty(
                    "display", "none", "important"
                );
                button.setAttribute("aria-hidden", "true");
                button.tabIndex = -1;
                count++;
            }
        }

        console.info(
            "Proposal-only UI: hidden alternatives =", count
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            keepProposalOnly,
            { once: true }
        );
    } else {
        keepProposalOnly();
    }

    window.addEventListener(
        "load",
        keepProposalOnly,
        { once: true }
    );
})();
