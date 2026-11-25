(() => {
    "use strict";

    function normalize(s) {
        return String(s || "").replace(/\s+/g, "");
    }

    function simplify() {
        const oldLabels = [
            "??????????",
            "??????????",
            "????????",
            "???????",
            "??????"
        ];

        for (const b of document.querySelectorAll("button")) {
            const label = normalize(b.textContent);
            if (oldLabels.some(x => label.includes(normalize(x)))) {
                b.hidden = true;
                b.style.setProperty("display", "none", "important");
                b.setAttribute("aria-hidden", "true");
                b.tabIndex = -1;
            }
        }

        // The question-search selector should show only the proposal.
        for (const select of document.querySelectorAll("select")) {
            const options = Array.from(select.options);
            const proposals = options.filter(o =>
                normalize(o.textContent).includes("????")
            );

            if (proposals.length !== 1 || options.length < 2) {
                continue;
            }

            const keep = proposals[0];
            for (const option of options) {
                if (option !== keep) option.remove();
            }
            select.value = keep.value;
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", simplify, {
            once: true
        });
    } else {
        simplify();
    }
    window.addEventListener("load", simplify, { once: true });
})();