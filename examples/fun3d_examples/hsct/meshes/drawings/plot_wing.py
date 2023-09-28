import numpy as np, matplotlib.pyplot as plt

LEchopFrac = 0.05
TEchopFrac = 0.08

area = 700
aspect = 7.0

phi_LE = [70, 50, 30]
phi_TE = [15, 10, 8]

span = np.sqrt(area * aspect)
print(f"span = {span}")
span_fr = [0.2, 0.3]
span_fr += [1 - sum(span_fr)]
sspan = 0.5 * span
sspans = [sspan * _ for _ in span_fr]
chord_drop = [
    sspans[i] * (np.tan(np.deg2rad(phi_LE[i])) + np.tan(np.deg2rad(phi_TE[i])))
    for i in range(3)
]
print(f"chord drops = {chord_drop}")
area_drop = [
    sspans[0] * 0.5 * chord_drop[0],
    0.5 * sspans[1] * (2 * chord_drop[0] + chord_drop[1]),
    0.5 * sspans[2] * (2 * chord_drop[0] + 2 * chord_drop[1] + chord_drop[2]),
]
chords = [(area + sum(area_drop)) / sspan]
for i in range(3):
    chords += [chords[i] - chord_drop[i]]
print(f"chords = {chords}")

xLE = [0.0]
for i in range(3):
    xLE += [xLE[i] + sspans[i] * np.tan(np.deg2rad(phi_LE[i]))]
xTE = [xLE[i] + chords[i] for i in range(4)]

z = [sum(sspans[:i]) for i in range(4)]

xperim = xLE + xTE[::-1] + [xLE[0]]
zperim = z + z[::-1] + [z[0]]

plt.plot(xperim, zperim, "k-", linewidth=2)

# plot spars
nspars = 3
nribs = 25

spar_vars = [1.0, 0.0]
spar_vars += [1 - sum(spar_vars)]
rib_vars = [1.0, 0.0]
rib_vars += [1 - sum(rib_vars)]

# plot spars
spar_colors = plt.cm.jet(np.linspace(0, 1, nribs - 2))
for ispar in range(1, nspars + 1):
    nom_spar_fr = ispar / (nspars + 1)
    spar_fr = np.sum(
        np.array(spar_vars)
        * np.array([nom_spar_fr, nom_spar_fr**2, nom_spar_fr**3])
    )
    xspar = [xLE[i] * (1 - spar_fr) + xTE[i] * spar_fr for i in range(4)]
    for ispan in range(3):
        plt.plot(
            xspar[ispan : ispan + 2],
            z[ispan : ispan + 2],
            linewidth=1,
            color=spar_colors[ispar - 1],
        )

# plot ribs
rib_colors = plt.cm.jet(np.linspace(0, 1, nribs - 2))
for imid in range(nribs - 2):
    irib = imid + 2
    nom_rib_fr = (irib - 1) / (nribs - 1)
    rib_fr = np.sum(
        np.array(spar_vars) * np.array([nom_rib_fr, nom_rib_fr**2, nom_rib_fr**3])
    )
    zrib = z[0] * (1 - rib_fr) + z[-1] * rib_fr
    if rib_fr < span_fr[0]:
        fr = rib_fr / span_fr[0]
        i = 0
    elif rib_fr < sum(span_fr[:2]):
        fr = (rib_fr - span_fr[0]) / (sum(span_fr[:2]) - span_fr[0])
        i = 1
    else:
        fr = (rib_fr - sum(span_fr[:2])) / (1 - sum(span_fr[:2]))
        i = 2
    print(f"irib = {irib}, ribfr = {rib_fr}, fr = {fr}")

    xrib = [xLE[i] * (1 - fr) + xLE[i + 1] * fr, xTE[i] * (1 - fr) + xTE[i + 1] * fr]

    plt.plot(xrib, [zrib, zrib], linewidth=1, color=rib_colors[imid])

# plot angles phi_i^(LE or TE)
LEcirc_dist = [3.0 for _ in range(3)]
TEcirc_dist = [6.0 for _ in range(3)]
for i in range(3):
    # LE vline
    plt.plot(
        [xLE[i], xLE[i]], [z[i], min(z[i] + sspan * 0.4, z[-1])], "k--", linewidth=1
    )

    # LE circle
    dx = xLE[i + 1] - xLE[i]
    dz = z[i + 1] - z[i]
    alpha0 = np.abs(np.arctan(dz / dx))
    thetas = np.linspace(alpha0, np.pi / 2, 40)
    dist = LEcirc_dist[i]
    plt.plot(xLE[i] + dist * np.cos(thetas), z[i] + dist * np.sin(thetas), "k--")
    # text_angle = alpha0+0.5*(np.pi/2-alpha0)
    # plt.text(xLE[i]-3+(dist+0.5)*np.cos(text_angle), z[i]+1+(dist+0.5)*np.sin(text_angle), f"phi{i}_LE", fontsize=8)

    # TE vline
    plt.plot(
        [xTE[i], xTE[i]], [z[i], min(z[i] + sspan * 0.4, z[-1])], "k--", linewidth=1
    )

    # TE circle
    dx = xTE[i + 1] - xTE[i]
    dz = z[i + 1] - z[i]
    alpha0 = np.abs(np.arctan(dz / dx))
    thetas = np.linspace(alpha0, np.pi / 2, 40)
    dist = TEcirc_dist[i]
    plt.plot(xTE[i] - dist * np.cos(thetas), z[i] + dist * np.sin(thetas), "k--")
    text_angle = alpha0 + 0.5 * (np.pi / 2 - alpha0)
    plt.text(
        xTE[i] + dist * np.cos(text_angle),
        z[i] + 1 + dist * np.sin(text_angle),
        f"phi{i}_TE",
        fontsize=8,
    )

    # hline between angles
    if i > 0:
        plt.plot([xLE[i], xTE[i]], [z[i], z[i]], "b--", linewidth=1)

plt.text(1, 3.5, "phi0_LE", fontsize=8)
plt.text(14, 9.5, "phi1_LE", fontsize=8)
plt.text(26, 20, "phi2_LE", fontsize=8)

plt.savefig("hsct_wing.png", dpi=300)
